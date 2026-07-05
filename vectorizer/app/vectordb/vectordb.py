import os
import sqlite3
import uuid
import re
import requests
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from vectorizer.app.core.settings import get_settings
from vectorizer.app.core.logger import logger
from .chunkenizer import recursive_character_splitting
from vectorizer.app.embeddings.embedding_generator import generate_embedding
from faq_extension.data_source import DataSourceManager
from faq_extension.document_parser import parse_document
import asyncio
import aiohttp
from tqdm.asyncio import tqdm_asyncio
from more_itertools import chunked
import time

settings = get_settings()

class VectorDB:
    def __init__(self, table_name, collection_name, create_collection=False):
        self.table_name = table_name
        self.collection_name = collection_name
        self.connect_to_qdrant()
        if create_collection:
            self.create_or_clear_collection()

    def connect_to_qdrant(self):
        # 如果提供了 API Key（云端 Qdrant），则使用认证；否则按本地方式直接连接
        logger.info(f"🔗 正在连接 Qdrant...")
        logger.info(f"🌐 URL: {settings.QDRANT_URL}")
        logger.info(f"🔑 API Key: {'***' + settings.QDRANT_KEY[-10:] if settings.QDRANT_KEY else '无'}")
        
        try:
            if settings.QDRANT_KEY:
                logger.info("🔐 使用 API Key 认证")
                self.client = QdrantClient(
                    url=settings.QDRANT_URL, 
                    api_key=settings.QDRANT_KEY,
                    timeout=60  # 将超时时间提高到 60 秒
                )
            else:
                logger.info("🔓 未提供 API Key，使用无认证方式连接")
                self.client = QdrantClient(url=settings.QDRANT_URL, timeout=60)
            
            # 测试连接
            logger.info("🧪 正在通过获取集合列表测试连接...")
            collections = self.client.get_collections()
            logger.info(f"✅ 已成功连接到 Qdrant！当前发现 {len(collections.collections)} 个已存在集合:")
            
            for collection in collections.collections:
                logger.info(f"  📁 {collection.name}")
                
        except Exception as e:
            logger.error(f"❌ 连接 Qdrant 失败，地址 {settings.QDRANT_URL}: {type(e).__name__}: {str(e)}")
            raise

    def create_or_clear_collection(self):
        max_retries = 3
        retry_delay = 5
        
        # 确定 embedding 维度
        embedding_size = self.get_embedding_dimensions()
        
        for attempt in range(max_retries):
            try:
                # 检查集合是否已存在
                exists = self.client.collection_exists(self.collection_name)
                
                if exists:
                    # 根据环境变量决定是否重建集合
                    should_recreate = settings.RECREATE_COLLECTIONS
                    if isinstance(should_recreate, str):
                        should_recreate = should_recreate.lower() == "true"
                    
                    if should_recreate:
                        logger.info(f"集合 {self.collection_name} 已存在，准备重建。")
                        self.client.delete_collection(collection_name=self.collection_name)
                        # 删除后稍作等待
                        import time
                        time.sleep(2)
                    else:
                        logger.info(f"集合 {self.collection_name} 已存在，跳过重建。")
                        return
                
                # 使用动态 embedding 维度创建新集合
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE)
                )
                logger.info(f"成功创建集合: {self.collection_name}，embedding 维度: {embedding_size}")
                return
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"第 {attempt + 1}/{max_retries} 次创建集合失败: {str(e)}")
                    logger.info(f"{retry_delay} 秒后重试...")
                    import time
                    time.sleep(retry_delay)
                else:
                    logger.error(f"重试 {max_retries} 次后仍未成功创建集合: {str(e)}")
                    raise
                    
    def get_embedding_dimensions(self):
        """根据当前使用的模型确定 embedding 维度"""
        try:
            if settings.USE_LOCAL_EMBEDDINGS:
                # 本地 embedding 直接从模型中读取维度
                from vectorizer.app.embeddings.local_embedding_generator import get_local_model
                model = get_local_model()
                embedding_size = model.get_sentence_embedding_dimension()
                logger.info(f"当前使用本地 embedding 模型，维度为 {embedding_size}")
                return embedding_size
            else:
                # 优先通过真实请求探测当前 API embedding 维度，避免与实际模型不一致
                test_embedding = generate_embedding("维度探测")
                embedding_size = len(test_embedding)
                logger.info(f"当前使用 API embedding 模型，维度为 {embedding_size}")
                return embedding_size
        except Exception as e:
            logger.warning(f"无法确定 embedding 维度: {str(e)}")
            logger.info("默认使用 1536 维")
            return 1536

    def format_content(self, data, collection_name):
        # 为不同集合实现对应的内容格式化逻辑
        if collection_name == 'car_rentals_collection':
            booking_status = "已预订" if data['booked'] else "未预订"
            return f"租车信息：{data['name']}，地点：{data['location']}，价格档位：{data['price_tier']}。" +\
                f"租赁开始日期为 {data['start_date']}，结束日期为 {data['end_date']}。" +\
                    f"当前租车状态：{booking_status}。"

        elif collection_name == 'excursions_collection':
            booking_status = "已预订" if data['booked'] else "未预订"
            return f"出游项目：{data['name']}，地点：{data['location']}。" +\
                f"补充说明：{data['details']}。" +\
                    f"当前出游状态：{booking_status}。" +\
                        f"关键词：{data['keywords']}。"

        elif collection_name == 'flights_collection':

            return f"航班 {data['flight_no']}，从 {data['departure_airport']} 飞往 {data['arrival_airport']}。" +\
                f"计划起飞时间为 {data['scheduled_departure']}，计划到达时间为 {data['scheduled_arrival']}。" +\
                    f"实际起飞时间为 {data['actual_departure']}，实际到达时间为 {data['actual_arrival']}。" +\
                        f"当前航班状态为“{data['status']}”，执飞机型代码为 {data['aircraft_code']}。"

        elif collection_name == 'hotels_collection':
            booking_status = "已预订" if data['booked'] else "未预订"
            return f"酒店 {data['name']}，位于 {data['location']}，价格档位为 {data['price_tier']}。" +\
                f"入住日期为 {data['checkin_date']}，退房日期为 {data['checkout_date']}。" +\
                    f"当前预订状态：{booking_status}。"

        elif collection_name == 'faq_collection':
            return data['page_content']  # FAQ 直接返回页面内容
        else:
            return str(data)

    def build_local_faq_entries(self, category, block_title, block_body, source):
        """将单个 FAQ 区块拆为可直接写库的问答条目。"""
        entries = []
        question_pattern = re.compile(r'(?m)^(\d+)\.\s+(.+?[？?])\s*$')
        question_matches = list(question_pattern.finditer(block_body))

        if not question_matches:
            content_parts = [f"## {category}"]
            if block_title:
                content_parts.append(f"### {block_title}")
            if block_body:
                content_parts.append(block_body)

            answer = block_body.strip()
            question = block_title or category
            entries.append(
                {
                    "page_content": "\n\n".join(part for part in content_parts if part).strip(),
                    "source": source,
                    "type": "faq",
                    "category": category,
                    "question": question,
                    "answer": answer,
                }
            )
            return entries

        block_prefix = block_body[:question_matches[0].start()].strip()

        for index, question_match in enumerate(question_matches):
            question_number = question_match.group(1).strip()
            question = question_match.group(2).strip()
            answer_start = question_match.end()
            answer_end = question_matches[index + 1].start() if index + 1 < len(question_matches) else len(block_body)
            answer = block_body[answer_start:answer_end].strip()

            content_parts = [f"## {category}"]
            if block_title:
                content_parts.append(f"### {block_title}")
            if index == 0 and block_prefix:
                content_parts.append(block_prefix)
            content_parts.append(f"{question_number}. {question}")
            if answer:
                content_parts.append(answer)

            entries.append(
                {
                    "page_content": "\n\n".join(part for part in content_parts if part).strip(),
                    "source": source,
                    "type": "faq",
                    "category": category,
                    "question": question,
                    "answer": answer,
                }
            )

        return entries

    def split_local_faq_entries(self, content, source):
        """按章节、小标题和问答拆分本地 FAQ 文档，提高检索命中精度。"""
        entries = []
        section_pattern = re.compile(r'(?ms)^##\s+(.+?)\n(.*?)(?=^##\s+|\Z)')
        subsection_pattern = re.compile(r'(?ms)^###\s+(.+?)\n(.*?)(?=^###\s+|\Z)')

        for section_match in section_pattern.finditer(content):
            category = section_match.group(1).strip()
            section_body = section_match.group(2).strip()
            subsection_matches = list(subsection_pattern.finditer(section_body))

            if subsection_matches:
                section_prefix = section_body[:subsection_matches[0].start()].strip()
                if section_prefix:
                    entries.extend(self.build_local_faq_entries(category, "", section_prefix, source))

                for subsection_match in subsection_matches:
                    block_title = subsection_match.group(1).strip()
                    block_body = subsection_match.group(2).strip()
                    entries.extend(self.build_local_faq_entries(category, block_title, block_body, source))
                continue

            entries.extend(self.build_local_faq_entries(category, "", section_body, source))

        if entries:
            return entries

        cleaned_content = content.strip()
        if not cleaned_content:
            return []

        return [
            {
                "page_content": cleaned_content,
                "source": source,
                "type": "faq",
                "category": "常见问题",
                "question": "常规 FAQ 信息",
                "answer": cleaned_content,
            }
        ]

    async def generate_embedding_async(self, content, session):
        max_retries = 5
        base_delay = 1
        
        # 修正 embedding API 的 base URL 组装方式
        base_url = settings.EMBEDDING_BASE_URL
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]  # 去掉结尾的 /v1
        embedding_url = f"{base_url}/v1/embeddings"
        
        # 使用配置中的 embedding 模型
        model = settings.EMBEDDING_MODEL
        
        logger.info(f"使用的 embedding URL: {embedding_url}")
        logger.info(f"使用的模型: {model}")
        logger.info(f"内容长度: {len(content)} 个字符")
        
        for attempt in range(max_retries):
            try:
                logger.info(f"第 {attempt + 1}/{max_retries} 次尝试，正在发起 embedding 请求...")
                
                headers = {"Authorization": f"Bearer {settings.EMBEDDING_API_KEY}"}
                payload = {"model": model, "input": content}
                
                async with session.post(
                    embedding_url,
                    headers=headers,
                    json=payload,
                    timeout=60  # 显式设置超时时间
                ) as response:
                    logger.info(f"响应状态码: {response.status}")
                    
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"HTTP 错误 {response.status}: {error_text}")
                        raise Exception(f"HTTP {response.status}: {error_text}")
                    
                    result = await response.json()
                    
                    if "data" in result and len(result["data"]) > 0:
                        embedding = result["data"][0]["embedding"]
                        logger.info(f"成功生成 embedding，维度数为 {len(embedding)}")
                        return embedding
                    else:
                        logger.error(f"API 返回结构不符合预期: {result}")
                        raise ValueError(f"API 返回结果不符合预期: {result}")
                        
            except Exception as e:
                logger.error(f"第 {attempt + 1} 次尝试出错: {type(e).__name__}: {str(e)}")
                
                if attempt == max_retries - 1:
                    logger.error(f"重试 {max_retries} 次后仍未成功生成 embedding")
                    raise
                    
                delay = base_delay * (2 ** attempt)
                logger.warning(f"{delay} 秒后重试...")
                await asyncio.sleep(delay)

    async def process_chunk(self, chunk, metadata, session):
        # 按 API 要求检查并限制输入长度（最大 2048 个字符）
        max_length = 2048
        original_length = len(chunk)
        
        if original_length > max_length:
            logger.warning(f"内容过长（{original_length} 个字符），截断至 {max_length} 个字符")
            chunk = chunk[:max_length]
            # 尽量避免从单词中间截断
            if chunk and not chunk[-1].isspace():
                last_space = chunk.rfind(' ')
                if last_space > max_length * 0.8:  # 仅在内容损失较少时回退到空格位置
                    chunk = chunk[:last_space]
        
        if len(chunk.strip()) == 0:
            logger.warning("处理后内容为空，跳过...")
            return None
            
        final_length = len(chunk)
        if final_length != original_length:
            logger.debug(f"文本长度已调整: {original_length} -> {final_length} 个字符")
            
        try:
            embedding = await self.generate_embedding_async(chunk, session)
            return PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "content": chunk,
                    "original_length": original_length,
                    "final_length": final_length,
                    **metadata
                }
            )
        except Exception as e:
            logger.error(f"处理分块失败（长度: {final_length}）: {str(e)}")
            raise

    async def create_embeddings_async(self):
        # 先测试 OpenAI 连接
        logger.info("🔍 正在执行预检查...")
        
        if not await self.test_openai_connection():
            raise Exception("OpenAI API 连接测试失败，无法继续生成 embedding。")
        
        logger.info("🚀 预检查通过，开始生成 embedding...")
        
        if self.table_name == "faq":
            await self.index_faq_docs()
        else:
            await self.index_regular_docs()

    async def index_regular_docs(self):
        logger.info(f"📊 正在处理普通集合: {self.collection_name}，来源表: {self.table_name}")
        
        try:
            # 检查数据库文件是否存在
            if not os.path.exists(settings.SQLITE_DB_PATH):
                logger.warning(f"⚠️ 未找到 SQLite 数据库文件: {settings.SQLITE_DB_PATH}")
                logger.info(f"💡 跳过集合 {self.collection_name}。请创建数据库文件以启用该集合。")
                return
                
            db_connection = sqlite3.connect(settings.SQLITE_DB_PATH)
            cursor = db_connection.cursor()
            
            # 检查数据表是否存在
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (self.table_name,))
            table_exists = cursor.fetchone() is not None
            
            if not table_exists:
                logger.warning(f"⚠️ 数据库中不存在表 '{self.table_name}'。")
                logger.info(f"💡 跳过集合 {self.collection_name}。请创建表 '{self.table_name}' 以启用该集合。")
                db_connection.close()
                return
            
            # 获取表数据
            cursor.execute(f"SELECT * FROM {self.table_name}")
            rows = cursor.fetchall()
            column_names = [column[0] for column in cursor.description]
            db_connection.close()
            
            if not rows:
                logger.warning(f"⚠️ 表 {self.table_name} 中没有数据。")
                logger.info(f"💡 跳过集合 {self.collection_name}。请向表 '{self.table_name}' 中写入数据以启用该集合。")
                return
                
            logger.info(f"📋 在表 {self.table_name} 中找到 {len(rows)} 条记录")
            
        except Exception as e:
            logger.error(f"❌ 表 {self.table_name} 发生数据库错误: {str(e)}")
            logger.info(f"💡 因数据库错误跳过集合 {self.collection_name}。")
            return

        # 将数据行转换为字典并处理内容
        data = [dict(zip(column_names, row)) for row in rows]
        
        # 按 FAQ 相同的文本长度规则处理每条数据
        processed_chunks = []
        chunk_metadata = []  # 保存每个分块对应的元数据
        max_chunk_size = 1900  # 预留保守缓冲，确保分块不会超过 2048
        
        for i, item in enumerate(data):
            try:
                # 使用已有的 format_content 方法格式化内容
                content = self.format_content(item, self.collection_name)
                logger.debug(f"第 {i+1} 条数据：格式化后的内容长度 = {len(content)} 个字符")
                
                if len(content) <= max_chunk_size:
                    processed_chunks.append(content)
                    chunk_metadata.append(item)  # 保存原始数据库记录
                else:
                    # 使用与 FAQ 相同的智能切分策略
                    logger.debug(f"内容过长（{len(content)} 个字符），开始智能切分...")
                    
                    # 先按段落切分
                    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
                    
                    current_chunk = ""
                    for paragraph in paragraphs:
                        if len(current_chunk) + len(paragraph) + 2 > max_chunk_size:
                            if current_chunk:
                                processed_chunks.append(current_chunk.strip())
                                chunk_metadata.append(item)  # 保存原始数据库记录
                                current_chunk = paragraph
                            else:
                                # 单个段落仍过长，则按句子切分
                                sentences = [s.strip() for s in re.split(r'[.!?]+', paragraph) if s.strip()]
                                for sentence in sentences:
                                    if not sentence.endswith(('.', '!', '?')):
                                        sentence += '.'
                                        
                                    if len(current_chunk) + len(sentence) + 1 > max_chunk_size:
                                        if current_chunk:
                                            processed_chunks.append(current_chunk.strip())
                                            chunk_metadata.append(item)  # 保存原始数据库记录
                                            current_chunk = sentence
                                        else:
                                            # 单句仍过长，则直接截断
                                            processed_chunks.append(sentence[:max_chunk_size])
                                            chunk_metadata.append(item)  # 保存原始数据库记录
                                    else:
                                        current_chunk = current_chunk + " " + sentence if current_chunk else sentence
                        else:
                            current_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
                    
                    # 补上最后一个分块
                    if current_chunk:
                        processed_chunks.append(current_chunk.strip())
                        chunk_metadata.append(item)  # 保存原始数据库记录
                        
            except Exception as e:
                logger.error(f"❌ 处理第 {i+1} 条数据时出错: {str(e)}")
                continue
        
        # 如有需要，再做一轮递归切分，同时保留元数据
        final_chunks = []
        final_metadata = []
        
        for chunk, metadata in zip(processed_chunks, chunk_metadata):
            if chunk.strip():
                split_chunks = recursive_character_splitting(chunk, chunk_size=1800, chunk_overlap=20)
                valid_split_chunks = [c for c in split_chunks if c.strip()]
                final_chunks.extend(valid_split_chunks)
                # 每个拆分后的分块沿用原分块的元数据
                final_metadata.extend([metadata] * len(valid_split_chunks))
        
        if not final_chunks:
            logger.warning(f"⚠️ {self.collection_name} 未生成任何有效分块")
            return
            
        # 校验所有分块是否都在长度限制内
        oversized_chunks = [i for i, chunk in enumerate(final_chunks) if len(chunk) > 2048]
        if oversized_chunks:
            logger.warning(f"⚠️ 发现 {len(oversized_chunks)} 个分块超过 2048 个字符，开始紧急截断...")
            for i in oversized_chunks:
                original_length = len(final_chunks[i])
                final_chunks[i] = final_chunks[i][:2000]
                logger.warning(f"  分块 {i}: 已从 {original_length} 截断为 {len(final_chunks[i])} 个字符")
        
        logger.info(f"📋 为 {self.collection_name} 生成了 {len(final_chunks)} 个有效分块")

        # 参照 FAQ 的异步流程按批处理分块
        batch_size = 50  # 更小的批次便于错误处理
        delay = 1  # 批次之间的等待秒数
        total_indexed = 0

        async with aiohttp.ClientSession() as session:
            for i in range(0, len(final_chunks), batch_size):
                batch = final_chunks[i:i+batch_size]
                batch_original_metadata = final_metadata[i:i+batch_size]
                logger.info(f"🔄 正在处理批次 {i//batch_size + 1}/{(len(final_chunks) + batch_size - 1)//batch_size}（{len(batch)} 个分块）")
                
                # 为每个分块构造元数据，包含原始数据库字段
                batch_metadata = []
                for j, original_meta in enumerate(batch_original_metadata):
                    combined_metadata = {
                        "type": self.table_name, 
                        "batch": i//batch_size + 1,
                        **original_meta  # 包含所有原始数据库字段
                    }
                    batch_metadata.append(combined_metadata)
                
                tasks = [self.process_chunk(chunk, metadata, session) for chunk, metadata in zip(batch, batch_metadata)]
                
                points = []
                for task in tqdm_asyncio.as_completed(tasks, desc=f"正在为 {self.collection_name} 生成 embedding（批次 {i//batch_size + 1}）", total=len(tasks)):
                    try:
                        point = await task
                        if point is not None:
                            points.append(point)
                    except Exception as e:
                        logger.error(f"❌ 处理分块时出错: {str(e)}")

                if points:
                    try:
                        self.client.upsert(
                            collection_name=self.collection_name,
                            points=points
                        )
                        logger.info(f"💾 已向 {self.collection_name} 写入 {len(points)} 条文档（批次 {i//batch_size + 1}）")
                        total_indexed += len(points)
                    except Exception as e:
                        logger.error(f"❌ 批量写入 Qdrant 时出错: {str(e)}")

                # 批次间稍作等待，避免触发限流
                if i + batch_size < len(final_chunks):
                    logger.debug(f"⏳ 下一批前等待 {delay} 秒...")
                    await asyncio.sleep(delay)

        logger.info(f"✅ {self.collection_name} 索引完成，共写入 {total_indexed} 条文档")

    async def index_faq_docs(self):
        local_docs = []
        data_source_manager = DataSourceManager()
        local_sources = data_source_manager.get_local_sources()

        for source_config in local_sources:
            files = data_source_manager.scan_source_files(source_config)
            for file_info in files:
                file_path = file_info["path"]
                content = parse_document(file_path)
                if not content or not content.strip():
                    logger.warning(f"本地 FAQ 文件内容为空，已跳过: {file_path}")
                    continue

                local_docs.append(
                    {
                        "page_content": content.strip(),
                        "source": file_path,
                    }
                )
                logger.info(f"已加载本地 FAQ 文件: {file_path}")

        if local_docs:
            logger.info("📄 优先使用本地 FAQ 知识库文件进行写入")
            initial_docs = []
            for doc in local_docs:
                initial_docs.extend(self.split_local_faq_entries(doc["page_content"], doc["source"]))
        else:
            faq_url = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/swiss_faq.md"

            logger.info(f"📄 正在下载 FAQ 内容，来源: {faq_url}")

            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(faq_url) as response:
                        logger.info(f"📈 FAQ 下载响应状态码: {response.status}")

                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(f"❌ FAQ 下载失败: HTTP {response.status} - {error_text}")
                            raise Exception(f"FAQ 下载失败: HTTP {response.status}")

                        faq_text = await response.text()
                        logger.info(f"📝 FAQ 内容下载完成，共 {len(faq_text)} 个字符")

                except Exception as e:
                    logger.error(f"💥 下载 FAQ 时出错: {str(e)}")
                    raise

            # 先按标题将 FAQ 拆成多个文档
            initial_docs = [
                {
                    "page_content": txt.strip(),
                    "source": faq_url,
                    "type": "faq",
                }
                for txt in re.split(r"(?=\n##)", faq_text) if txt.strip()
            ]

        logger.info(f"📋 FAQ 初步拆分为 {len(initial_docs)} 个部分")
        
        # 对较大的文档继续切分，保证不超过 2048 字符限制
        max_chunk_size = 1900  # 预留更保守的缓冲，确保文档不超过 2048
        docs = []
        
        for i, initial_doc in enumerate(initial_docs):
            doc_content = initial_doc["page_content"]
            doc_metadata = {key: value for key, value in initial_doc.items() if key != "page_content"}
            logger.debug(f"正在处理第 {i+1} 个部分: {len(doc_content)} 个字符")
            
            if len(doc_content) <= max_chunk_size:
                docs.append({"page_content": doc_content, **doc_metadata})
            else:
                logger.info(f"第 {i+1} 个部分过长（{len(doc_content)} 个字符），开始智能切分...")
                
                # 先按段落切分大文档
                paragraphs = [p.strip() for p in doc_content.split('\n\n') if p.strip()]
                logger.debug(f"  已拆分为 {len(paragraphs)} 个段落")
                
                current_chunk = ""
                chunk_count = 0
                
                for j, paragraph in enumerate(paragraphs):
                    # 如果加上该段后会超限，就先保存当前分块再开始新分块
                    if len(current_chunk) + len(paragraph) + 2 > max_chunk_size:  # +2 for \n\n
                        if current_chunk:
                            docs.append({"page_content": current_chunk.strip(), **doc_metadata})
                            chunk_count += 1
                            logger.debug(f"    已创建分块 {chunk_count}: {len(current_chunk)} 个字符")
                            current_chunk = paragraph
                        else:
                            # 单个段落本身过长，则按句子切分
                            logger.debug(f"    第 {j+1} 个段落过长（{len(paragraph)} 个字符），按句子切分...")
                            sentences = [s.strip() for s in re.split(r'[.!?]+', paragraph) if s.strip()]
                            
                            for k, sentence in enumerate(sentences):
                                # 如果标点被切掉，则补回去
                                if k < len(sentences) - 1 or not sentence.endswith(('.', '!', '?')):
                                    sentence += '.'
                                    
                                if len(current_chunk) + len(sentence) + 1 > max_chunk_size:
                                    if current_chunk:
                                        docs.append({"page_content": current_chunk.strip(), **doc_metadata})
                                        chunk_count += 1
                                        logger.debug(f"      已创建句子分块 {chunk_count}: {len(current_chunk)} 个字符")
                                        current_chunk = sentence
                                    else:
                                        # 单句仍过长，则按单词切分
                                        logger.debug(f"      句子过长（{len(sentence)} 个字符），按单词切分...")
                                        words = sentence.split()
                                        word_chunk = ""
                                        
                                        for word in words:
                                            if len(word_chunk) + len(word) + 1 > max_chunk_size:
                                                if word_chunk:
                                                    docs.append({"page_content": word_chunk.strip(), **doc_metadata})
                                                    chunk_count += 1
                                                    logger.debug(f"        已创建单词分块 {chunk_count}: {len(word_chunk)} 个字符")
                                                    word_chunk = word
                                                else:
                                                    # 单个单词仍过长，则直接截断（极少见）
                                                    truncated = word[:max_chunk_size]
                                                    docs.append({"page_content": truncated, **doc_metadata})
                                                    chunk_count += 1
                                                    logger.warning(f"        超长单词已截断: {len(word)} -> {len(truncated)} 个字符")
                                            else:
                                                word_chunk = word_chunk + " " + word if word_chunk else word
                                        
                                        if word_chunk:
                                            current_chunk = word_chunk
                                else:
                                    current_chunk = current_chunk + " " + sentence if current_chunk else sentence
                    else:
                        current_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
                
                # 若最后仍有内容，则补上最后一个分块
                if current_chunk:
                    docs.append({"page_content": current_chunk.strip(), **doc_metadata})
                    chunk_count += 1
                    logger.debug(f"    最后一个分块 {chunk_count}: {len(current_chunk)} 个字符")
                
                logger.info(f"  第 {i+1} 个部分已拆分为 {chunk_count} 个分块")
        
        logger.info(f"📋 FAQ 最终拆分为 {len(docs)} 个文档（已做严格长度优化）")
        
        # 记录文档长度统计并做最终校验
        if docs:
            doc_lengths = [len(doc["page_content"]) for doc in docs]
            logger.info(f"📏 文档长度统计: 最小={min(doc_lengths)}，最大={max(doc_lengths)}，平均={sum(doc_lengths)//len(doc_lengths)}")
            
            sample_doc = docs[0]["page_content"][:200] + "..." if len(docs[0]["page_content"]) > 200 else docs[0]["page_content"]
            logger.info(f"📖 示例文档: {sample_doc}")
            
            # 检查是否仍有超长文档，并进行修正
            oversized_docs = [i for i, doc in enumerate(docs) if len(doc["page_content"]) > 2048]
            if oversized_docs:
                logger.warning(f"⚠️ 发现 {len(oversized_docs)} 个文档仍超过 2048 个字符，开始紧急截断...")
                for i in oversized_docs:
                    original_length = len(docs[i]["page_content"])
                    docs[i]["page_content"] = docs[i]["page_content"][:2000]  # 紧急截断
                    logger.warning(f"  文档 {i}: 已从 {original_length} 截断为 {len(docs[i]['page_content'])} 个字符")
                
                # 复核修正后是否仍有超长文档
                final_oversized = [i for i, doc in enumerate(docs) if len(doc["page_content"]) > 2048]
                if final_oversized:
                    logger.error(f"❌ 严重错误：紧急修正后仍有 {len(final_oversized)} 个文档超过 2048 个字符！")
                else:
                    logger.info(f"✅ 紧急修正后，所有文档都已控制在 2048 字符以内")
            else:
                logger.info(f"✅ 所有文档都在 2048 字符限制以内！")

        logger.info(f"🤖 开始为 {len(docs)} 个 FAQ 文档生成 embedding（均保证不超过 2048 字符）...")
        
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.process_chunk(
                    doc["page_content"],
                    {key: value for key, value in doc.items() if key != "page_content"},
                    session
                )
                for doc in docs
            ]

            try:
                points = await tqdm_asyncio.gather(*tasks, desc="正在为 FAQ 文档生成 embedding")
                logger.info(f"✅ 成功生成 {len([p for p in points if p is not None])} 个 embedding")
            except Exception as e:
                logger.error(f"💥 生成 embedding 过程中出错: {str(e)}")
                raise

        if points:
            logger.info(f"📁 正在向 Qdrant 集合 {self.collection_name} 写入 {len(points)} 个点")
            
            try:
                for batch in chunked(points, 100):  # 可根据需要调整批次大小
                    non_null_batch = [p for p in batch if p is not None]
                    if non_null_batch:
                        logger.info(f"📎 正在批量写入 {len(non_null_batch)} 个点...")
                        self.client.upsert(
                            collection_name=self.collection_name,
                            points=non_null_batch
                        )
                        
                logger.info(f"✅ 已成功将 {len([p for p in points if p is not None])} 个 FAQ 文档写入 {self.collection_name}。")
            except Exception as e:
                logger.error(f"💥 写入 Qdrant 时出错: {str(e)}")
                raise
        else:
            logger.warning("⚠️ 没有任何 FAQ 文档成功生成 embedding 并写入索引。")

    def create_embeddings(self):
        asyncio.run(self.create_embeddings_async())

    async def test_openai_connection(self):
        """使用当前配置的 embedding 模型测试 API 连接。"""
        logger.info("正在测试 OpenAI API 连接...")
        
        # 修正 embedding API 的 base URL 组装方式
        base_url = settings.EMBEDDING_BASE_URL
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]  # 去掉结尾的 /v1
        embedding_url = f"{base_url}/v1/embeddings"
        
        logger.info(f"基础 URL: {base_url}")
        logger.info(f"Embedding 接口 URL: {embedding_url}")
        
        test_content = "你好，这是一条测试文本。"
        available_models = await self.get_available_models()

        models_to_try = [settings.EMBEDDING_MODEL]
        for model in available_models:
            if model not in models_to_try:
                models_to_try.append(model)

        logger.info(f"准备测试 {len(models_to_try)} 个模型，优先使用当前配置模型: {settings.EMBEDDING_MODEL}")

        for i, model in enumerate(models_to_try):
            try:
                logger.info(f"[{i+1}/{len(models_to_try)}] 正在测试模型: {model}")
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        embedding_url,
                        headers={"Authorization": f"Bearer {settings.EMBEDDING_API_KEY}"},
                        json={"model": model, "input": test_content},
                        timeout=30
                    ) as response:
                        logger.info(f"模型 {model} 的响应状态码: {response.status}")
                        
                        if response.status == 200:
                            result = await response.json()
                            if "data" in result and len(result["data"]) > 0:
                                embedding = result["data"][0]["embedding"]
                                logger.info(f"成功！模型 {model} 可用，embedding 维度: {len(embedding)}")
                                self.working_model = model
                                return True
                        
                        error_text = await response.text()
                        logger.warning(f"模型 {model} 测试失败: HTTP {response.status} - {error_text[:100]}...")
                        
            except Exception as e:
                logger.warning(f"测试模型 {model} 时出错: {type(e).__name__}: {str(e)}")
                continue
        
        logger.error(f"已测试的 {len(models_to_try)} 个 embedding 模型全部失败，当前配置的 embedding 服务不可用。")
        return False

    async def get_available_models(self):
        """获取 API 当前可用的模型列表"""
        base_url = settings.EMBEDDING_BASE_URL
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        models_url = f"{base_url}/v1/models"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    models_url,
                    headers={"Authorization": f"Bearer {settings.EMBEDDING_API_KEY}"},
                    timeout=30
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "data" in result:
                            models = [model["id"] for model in result["data"]]
                            embedding_models = [m for m in models if "embedding" in m.lower()]
                            logger.info(f"当前可用的 embedding 模型: {embedding_models}")
                            return embedding_models
                    
                    error_text = await response.text()
                    logger.warning(f"获取模型列表失败: HTTP {response.status} - {error_text}")
                    return []
        except Exception as e:
            logger.warning(f"获取模型列表时出错: {str(e)}")
            return []

    def search(self, query, limit=2, with_payload=True):
        query_vector = generate_embedding(query)
        if hasattr(self.client, "search"):
            return self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=with_payload
            )

        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=with_payload
        )
        return search_result.points

if __name__ == "__main__":
    vectordb = VectorDB("example_table", "example_collection")
    vectordb.create_embeddings()
