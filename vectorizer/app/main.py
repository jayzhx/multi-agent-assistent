from vectorizer.app.core.logger import logger
from vectorizer.app.vectordb.vectordb import VectorDB
from vectorizer.app.core.settings import get_settings

settings = get_settings()

def create_collections():
    # 定义所有可用的集合
    # FAQ 集合使用外部数据，其余集合依赖 SQLite 数据库
    collections = [
        ("faq", "faq_collection"),                 # 使用网页外部 FAQ 数据，始终可用
        ("flights", "flights_collection"),         # 需要包含 flights 表的 SQLite 数据库
        ("hotels", "hotels_collection"),           # 需要包含 hotels 表的 SQLite 数据库
        ("car_rentals", "car_rentals_collection"), # 需要包含 car_rentals 表的 SQLite 数据库
        ("trip_recommendations", "excursions_collection")    # 需要包含 trip_recommendations 表的 SQLite 数据库
    ]
    
    logger.info(f"向量化程序启动，本次需要处理 {len(collections)} 个集合")
    logger.info(f"集合列表: {[f'{table}->{collection}' for table, collection in collections]}")
    
    # 先测试 embedding API 连接
    logger.info("正在测试 embedding API 连接...")
    test_vectordb = VectorDB("test", "test_collection", create_collection=False)
    import asyncio
    connection_ok = asyncio.run(test_vectordb.test_openai_connection())
    
    if not connection_ok:
        logger.error("Embedding API 连接失败，请检查配置。")
        logger.error("请参考 EMBEDDING_SETUP.md 中的配置说明。")
        return
    
    # 记录成功和失败的集合
    successful_collections = []
    failed_collections = []

    for table_name, collection_name in collections:
        try:
            logger.info(f"\n" + "="*80)
            logger.info(f"正在处理: {table_name} -> {collection_name}")
            logger.info(f"="*80)
            
            logger.info(f"正在为 {table_name} 启动向量数据库服务")
            vectordb = VectorDB(table_name=table_name, collection_name=collection_name, create_collection=True)
            
            logger.info(f"开始为 {collection_name} 生成 embedding...")
            vectordb.create_embeddings()
            
            logger.info(f"✅ {collection_name} 的 embedding 生成与存储已完成")
            successful_collections.append((table_name, collection_name))
            
        except Exception as e:
            logger.error(f"❌ 处理 {table_name} 时发生错误: {type(e).__name__}: {str(e)}")
            logger.exception("详细错误信息:")
            failed_collections.append((table_name, collection_name, str(e)))
            
            # 对依赖数据库的集合输出提示信息
            if table_name != "faq":
                logger.info(f"💡 提示: {table_name} 集合依赖包含 {table_name} 表的 SQLite 数据库。"
                           f"如果数据库为空或缺失，将跳过该集合。")
    
    # 汇总报告
    logger.info(f"\n" + "="*80)
    logger.info(f"向量化结果汇总")
    logger.info(f"="*80)
    
    if successful_collections:
        logger.info(f"✅ 成功处理 {len(successful_collections)} 个集合:")
        for table, collection in successful_collections:
            logger.info(f"   • {table} -> {collection}")
    
    if failed_collections:
        logger.warning(f"❌ 共有 {len(failed_collections)} 个集合处理失败:")
        for table, collection, error in failed_collections:
            logger.warning(f"   • {table} -> {collection}: {error[:100]}...")
    
    logger.info(f"\n🎯 系统已就绪，当前可用集合数为 {len(successful_collections)}")
    
    if not successful_collections:
        logger.error(f"⚠️ 没有任何集合成功创建，请检查配置和数据库设置。")

if __name__ == "__main__":
    create_collections()
    logger.info(f"\n🚀 提示: 若要填充其他集合（flights、hotels、car_rentals、excursions），"
               f"请确保位于 {settings.SQLITE_DB_PATH} 的 SQLite 数据库中包含所需数据表及数据。")
