# customer_support_chat/app/services/tools/woocommerce.py

import httpx
from langchain_core.tools import tool
from customer_support_chat.app.core.settings import get_settings
from customer_support_chat.app.core.logger import logger
from typing import List, Dict, Optional

settings = get_settings()

@tool
def search_products(query: str, limit: int = 10) -> List[Dict]:
    """根据查询条件在 WooCommerce 中搜索商品。
    
    Args:
        query: 搜索条件（例如商品名、分类）。
        limit: 最多返回的商品数量（默认 10）。
        
    Returns:
        包含关键信息的商品字典列表。
    """
    logger.info(f"🔍 调用 WooCommerce 商品搜索，query: '{query}'，limit: {limit}")
    
    if not settings.WOOCOMMERCE_API_URL or not settings.WOOCOMMERCE_CONSUMER_KEY or not settings.WOOCOMMERCE_CONSUMER_SECRET:
        error_msg = "WooCommerce API 凭据未配置。"
        logger.error(f"❌ {error_msg}")
        raise ValueError(error_msg)
    
    # 确保 URL 符合正确的 WooCommerce REST API 格式
    base_url = settings.WOOCOMMERCE_API_URL.rstrip('/')
    logger.info(f"🌐 配置中的基础 URL: {base_url}")
    
    if not base_url.endswith('/wp-json/wc/v3'):
        # 如果 URL 中未包含 API 路径，则自动补上
        if '/wp-json/wc/v3' not in base_url:
            url = f"{base_url}/wp-json/wc/v3/products"
        else:
            url = f"{base_url}/products"
    else:
        url = f"{base_url}/products"
    
    logger.info(f"🌍 最终 API URL: {url}")
    params = {
        "search": query,
        "per_page": min(limit, 100)  # WooCommerce API 的数量上限
    }
    
    logger.info(f"📦 请求参数: {params}")
    
    # 使用同步 httpx 客户端
    with httpx.Client(verify=False, timeout=30.0) as client:  # 本地开发时禁用 SSL 校验
        try:
            logger.info(f"🚀 正在向以下地址发起 API 请求: {url}")
            response = client.get(
                url,
                params=params,
                auth=httpx.BasicAuth(settings.WOOCOMMERCE_CONSUMER_KEY, settings.WOOCOMMERCE_CONSUMER_SECRET)
            )
            
            logger.info(f"📊 响应状态码: {response.status_code}")
            response.raise_for_status()
            products = response.json()
            
            logger.info(f"✅ 成功获取 {len(products)} 个商品")
            
            # 提取关键信息
            simplified_products = []
            for product in products:
                simplified_products.append({
                    "id": product.get("id"),
                    "name": product.get("name"),
                    "price": product.get("price"),
                    "description": product.get("short_description") or product.get("description", "")[:100] + "...",
                    "permalink": product.get("permalink"),
                    "sku": product.get("sku"),
                })
            
            return simplified_products
        except httpx.HTTPStatusError as e:
            raise Exception(f"搜索商品时发生 HTTP 错误：{e}（状态码：{e.response.status_code}）")
        except httpx.TimeoutException as e:
            raise Exception(f"搜索商品时发生超时。WooCommerce 服务器可能较慢或不可用：{e}")
        except httpx.ConnectError as e:
            raise Exception(f"搜索商品时发生连接错误。请检查 WooCommerce 服务器是否正在运行：{e}")
        except Exception as e:
            raise Exception(f"搜索商品时发生错误：{e}")

@tool
def search_orders(search_type: str, search_value: str, limit: int = 10) -> List[Dict]:
    """根据指定条件在 WooCommerce 中搜索订单。
    
    参数:
        search_type: 搜索类型，必须是 'email'、'name' 或 'id' 之一。
        search_value: 与 search_type 对应的搜索值。
        limit: 最多返回的订单数量（默认 10）。
        
    返回:
        包含关键信息的订单字典列表。
    """
    logger.info(f"🔍 调用 WooCommerce 订单搜索，search_type: '{search_type}'，search_value: '{search_value}'，limit: {limit}")
    
    if not settings.WOOCOMMERCE_API_URL or not settings.WOOCOMMERCE_CONSUMER_KEY or not settings.WOOCOMMERCE_CONSUMER_SECRET:
        error_msg = "WooCommerce API 凭据未配置。"
        logger.error(f"❌ {error_msg}")
        raise ValueError(error_msg)
    
    # 校验 search_type
    valid_search_types = ['email', 'name', 'id']
    if search_type not in valid_search_types:
        error_msg = f"无效的 search_type：{search_type}。必须是以下之一：{valid_search_types}"
        logger.error(f"❌ {error_msg}")
        raise ValueError(error_msg)
    
    # 确保 URL 符合正确的 WooCommerce REST API 格式
    base_url = settings.WOOCOMMERCE_API_URL.rstrip('/')
    logger.info(f"🌐 配置中的基础 URL: {base_url}")
    
    if not base_url.endswith('/wp-json/wc/v3'):
        # 如果 URL 中未包含 API 路径，则自动补上
        if '/wp-json/wc/v3' not in base_url:
            url = f"{base_url}/wp-json/wc/v3/orders"
        else:
            url = f"{base_url}/orders"
    else:
        url = f"{base_url}/orders"
    
    logger.info(f"🌍 订单接口最终 API URL: {url}")
    
    # 根据搜索类型构造参数
    params = {
        "per_page": min(limit, 100)  # WooCommerce API 的数量上限
    }
    
    if search_type == 'email':
        # 按客户邮箱搜索
        params["customer_email"] = search_value
    elif search_type == 'name':
        # 按姓名搜索时，通过账单名和姓进行模糊匹配
        params["search"] = search_value
    elif search_type == 'id':
        # 按 ID 搜索时，可以直接请求指定订单
        try:
            order_id = int(search_value)
            url = f"{url}/{order_id}"
            params = {}  # 查询指定订单时不需要额外参数
        except ValueError:
            # 如果不是有效整数，则按普通搜索处理
            params["search"] = search_value
    
    logger.info(f"📦 订单请求参数: {params}")
    logger.info(f"🔗 请求 URL: {url}")
    
    # 订单查询使用同步 httpx 客户端，并设置更长超时时间
    with httpx.Client(verify=False, timeout=60.0) as client:  # 为订单查询增加超时时间
        try:
            logger.info(f"🚀 正在向订单接口发起 API 请求: {url}")
            response = client.get(
                url,
                params=params,
                auth=httpx.BasicAuth(settings.WOOCOMMERCE_CONSUMER_KEY, settings.WOOCOMMERCE_CONSUMER_SECRET)
            )
            
            logger.info(f"📊 订单搜索响应状态码: {response.status_code}")
            response.raise_for_status()
            
            # 处理单个订单的响应（按 ID 查询时）
            if search_type == 'id' and params == {}:
                order = response.json()
                orders = [order] if order else []
            else:
                orders = response.json()
            
            logger.info(f"✅ 成功获取 {len(orders)} 个订单")
            
            # 记录搜索结果，便于调试
            if len(orders) == 0:
                logger.warning(f"⚠️ 未找到 search_type 为 {search_type}、search_value 为 '{search_value}' 的订单。")
            
            # 提取关键信息
            simplified_orders = []
            for order in orders:
                simplified_orders.append({
                    "id": order.get("id"),
                    "status": order.get("status"),
                    "total": order.get("total"),
                    "currency": order.get("currency"),
                    "customer_note": order.get("customer_note"),
                    "date_created": order.get("date_created"),
                    "billing": {
                        "first_name": order.get("billing", {}).get("first_name"),
                        "last_name": order.get("billing", {}).get("last_name"),
                        "email": order.get("billing", {}).get("email"),
                    },
                })
            
            return simplified_orders
        except httpx.HTTPStatusError as e:
            raise Exception(f"搜索订单时发生 HTTP 错误：{e}（状态码：{e.response.status_code}）")
        except httpx.TimeoutException as e:
            raise Exception(f"搜索订单时发生超时。WooCommerce 服务器可能较慢或不可用：{e}")
        except httpx.ConnectError as e:
            raise Exception(f"搜索订单时发生连接错误。请检查 WooCommerce 服务器是否正在运行：{e}")
        except Exception as e:
            raise Exception(f"搜索订单时发生错误：{e}")

# 如有需要，可在此处继续扩展 WooCommerce 相关工具
