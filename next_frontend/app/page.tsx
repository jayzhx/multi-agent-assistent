"use client";

import { FormEvent, useEffect, useState } from "react";

type ChatHistoryItem = {
  timestamp?: string;
  user_message: string;
  ai_response: string;
};

type LogEntry = {
  type: string;
  title: string;
  content: string;
  timestamp: string;
};

type ToolCall = {
  id: string;
  name: string;
  args: Record<string, unknown>;
};

type PendingAction = {
  tool_calls: ToolCall[];
  timestamp: number;
  approval_summary?: string;
};

type AuthUser = {
  id: number;
  username: string;
  passenger_id: string;
  created_at: string;
};

type SessionPayload = {
  session_id: string;
  user?: AuthUser;
  chat_history: ChatHistoryItem[];
  approvals_enabled: boolean;
};

type AuthPayload = {
  authenticated?: boolean;
  user?: AuthUser;
  error?: string;
};

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

type OrderSummary = {
  id: number;
  order_no: string;
  order_type: "flight" | "hotel" | "car" | "trip";
  total_amount_minor: number;
  currency: string;
  status: string;
  supplier_confirmation_no?: string;
  service_start_at?: string;
  service_end_at?: string;
  created_at: string;
};

type OrderDetail = OrderSummary & {
  supplier_name: string;
  updated_at: string;
  confirmed_at?: string;
  cancelled_at?: string;
  detail: Record<string, unknown> | null;
  segments: Record<string, unknown>[];
  passengers: Record<string, unknown>[];
  status_history: Record<string, unknown>[];
  supplier_attempts: Record<string, unknown>[];
};

type OrderPagination = {
  total: number;
  page: number;
  page_size: number;
};

type FeishuReadiness = {
  enabled: boolean;
  missing: string[];
  token_valid: boolean;
  contact_access: boolean;
  approval_definition_access: boolean | null;
  required_scopes: string[];
};

const apiBaseUrl =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8001";

function buildChatMessages(chatHistory: ChatHistoryItem[]): ChatMessage[] {
  return chatHistory.flatMap((item) => [
    { role: "user", content: item.user_message },
    { role: "assistant", content: item.ai_response },
  ]);
}

function formatTimeLabel(value?: string): string {
  if (!value) {
    return "--:--";
  }

  return new Date(value).toLocaleTimeString("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatDateTime(value: unknown): string {
  if (!value) {
    return "-";
  }
  return new Date(String(value)).toLocaleString("zh-CN");
}

function getOrderDetailRows(order: OrderDetail): Array<[string, string]> {
  const detail = order.detail ?? {};
  if (order.order_type === "flight") {
    const segment = order.segments[0] ?? {};
    return [
      ["航班号", String(segment.flight_no ?? "-")],
      ["航线", `${String(segment.departure_airport ?? "-")} → ${String(segment.arrival_airport ?? "-")}`],
      ["起飞时间", formatDateTime(segment.departure_at)],
      ["到达时间", formatDateTime(segment.arrival_at)],
      ["乘机人", order.passengers.map((item) => String(item.passenger_name ?? "")).filter(Boolean).join("、") || "-"],
    ];
  }
  if (order.order_type === "hotel") {
    return [
      ["酒店", String(detail.hotel_name ?? "-")],
      ["房型", String(detail.room_type ?? "-")],
      ["入住日期", String(detail.checkin_date ?? "-")],
      ["退房日期", String(detail.checkout_date ?? "-")],
      ["入住人", Array.isArray(detail.guest_names) ? detail.guest_names.join("、") : "-"],
    ];
  }
  if (order.order_type === "car") {
    return [
      ["车辆", String(detail.product_name ?? "-")],
      ["取车地点", String(detail.pickup_location ?? "-")],
      ["还车地点", String(detail.return_location ?? "-")],
      ["取车时间", formatDateTime(detail.pickup_at)],
      ["还车时间", formatDateTime(detail.return_at)],
      ["驾驶人", String(detail.driver_name ?? "-")],
    ];
  }
  return [
    ["行程", String(detail.product_name ?? "-")],
    ["出行日期", String(detail.visit_date ?? "-")],
    ["参与人数", String(detail.participant_count ?? "-")],
  ];
}

export default function HomePage() {
  const [sessionId, setSessionId] = useState("");
  const [currentUser, setCurrentUser] = useState<AuthUser | null>(null);
  const [chatHistory, setChatHistory] = useState<ChatHistoryItem[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [pendingAction, setPendingAction] = useState<PendingAction | null>(null);
  const [approvalsEnabled, setApprovalsEnabled] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [isBooting, setIsBooting] = useState(true);
  const [errorText, setErrorText] = useState("");
  const [authMode, setAuthMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [passengerId, setPassengerId] = useState("");
  const [isAuthenticating, setIsAuthenticating] = useState(false);
  const [orders, setOrders] = useState<OrderSummary[]>([]);
  const [isDeciding, setIsDeciding] = useState(false);
  const [orderPagination, setOrderPagination] = useState<OrderPagination>({ total: 0, page: 1, page_size: 5 });
  const [orderStatusFilter, setOrderStatusFilter] = useState("");
  const [orderTypeFilter, setOrderTypeFilter] = useState("");
  const [selectedOrder, setSelectedOrder] = useState<OrderDetail | null>(null);
  const [isOrderBusy, setIsOrderBusy] = useState(false);
  const [editStartDate, setEditStartDate] = useState("");
  const [editEndDate, setEditEndDate] = useState("");
  const [editFlightId, setEditFlightId] = useState("");
  const [editParticipantCount, setEditParticipantCount] = useState("1");
  const [feishuReadiness, setFeishuReadiness] = useState<FeishuReadiness | null>(null);

  const toolLogs = logs.filter(
    (entry) =>
      entry.type === "tool_call" ||
      entry.type === "tool_result" ||
      entry.type === "system_message",
  );
  const recentHistory = [...chatHistory].reverse().slice(0, 6);
  const recentToolLogs = [...toolLogs].reverse().slice(0, 6);
  const latestAssistantMessage =
    [...messages].reverse().find((message) => message.role === "assistant")
      ?.content ?? "";
  const statusText = isBooting
    ? "正在连接后端会话..."
    : !currentUser
      ? "请先登录，系统才会把机票、酒店和租车记录绑定到当前用户。"
      : isSending
        ? "助手正在处理你的请求..."
        : approvalsEnabled
          ? "审批模式已开启，可在左侧查看待处理动作。"
          : "当前为只读模式，可继续体验查询与推荐链路。";
  const modeLabel = approvalsEnabled ? "审批模式" : "只读模式";
  const feishuStatusText = !feishuReadiness
    ? "正在检测"
    : [
        feishuReadiness.token_valid ? "令牌有效" : "令牌不可用",
        feishuReadiness.contact_access ? "通讯录权限正常" : "待开通通讯录权限",
        feishuReadiness.approval_definition_access === null
          ? "待配置审批定义"
          : feishuReadiness.approval_definition_access
            ? "审批定义可访问"
            : "审批定义不可访问",
        feishuReadiness.enabled ? "飞书已启用" : "飞书未启用",
      ].join("，");

  useEffect(() => {
    void bootstrapAuth();
  }, []);

  useEffect(() => {
    if (!currentUser) {
      return;
    }

    const logTimer = window.setInterval(() => {
      void Promise.all([refreshOperationLog(), refreshOrders()]);
    }, 10000);

    const pendingTimer = window.setInterval(() => {
      void refreshPendingAction();
    }, 5000);

    const feishuTimer = window.setInterval(() => {
      void refreshFeishuReadiness();
    }, 60000);

    return () => {
      window.clearInterval(logTimer);
      window.clearInterval(pendingTimer);
      window.clearInterval(feishuTimer);
    };
  }, [currentUser]);

  useEffect(() => {
    if (currentUser) {
      void refreshOrders();
    }
  }, [currentUser, orderStatusFilter, orderTypeFilter, orderPagination.page]);

  function resetWorkspace() {
    setSessionId("");
    setChatHistory([]);
    setMessages([]);
    setLogs([]);
    setPendingAction(null);
    setApprovalsEnabled(false);
    setDraft("");
    setOrders([]);
    setOrderPagination({ total: 0, page: 1, page_size: 5 });
    setOrderStatusFilter("");
    setOrderTypeFilter("");
    setSelectedOrder(null);
    setFeishuReadiness(null);
  }

  async function bootstrapAuth() {
    try {
      const response = await fetch(`${apiBaseUrl}/auth/me`, {
        credentials: "include",
      });
      const data = (await response.json()) as AuthPayload;

      if (data.authenticated && data.user) {
        setCurrentUser(data.user);
        await Promise.all([
          bootstrapSession(),
          refreshOperationLog(),
          refreshPendingAction(),
          refreshOrders(),
          refreshFeishuReadiness(),
        ]);
        setErrorText("");
      } else {
        setCurrentUser(null);
        resetWorkspace();
      }
    } catch (error) {
      console.error("bootstrap auth error", error);
      setErrorText("当前无法连接后端，请确认 FastAPI 服务已经启动。");
    } finally {
      setIsBooting(false);
    }
  }

  async function bootstrapSession() {
    try {
      const response = await fetch(`${apiBaseUrl}/session`, {
        credentials: "include",
      });

      if (response.status === 401) {
        setCurrentUser(null);
        resetWorkspace();
        return;
      }

      const data = (await response.json()) as SessionPayload;
      const nextChatHistory = data.chat_history ?? [];
      setSessionId(data.session_id ?? "");
      setCurrentUser(data.user ?? currentUser);
      setChatHistory(nextChatHistory);
      setMessages(buildChatMessages(nextChatHistory));
      setApprovalsEnabled(Boolean(data.approvals_enabled));
      setErrorText("");
    } catch (error) {
      console.error("bootstrap session error", error);
      setErrorText("当前无法连接后端，请确认 FastAPI 服务已经启动。");
    }
  }

  async function refreshOperationLog() {
    if (!currentUser) {
      return;
    }

    try {
      const response = await fetch(`${apiBaseUrl}/operation-log`, {
        credentials: "include",
      });
      if (response.status === 401) {
        setCurrentUser(null);
        resetWorkspace();
        return;
      }

      const data = await response.json();
      if (!data.error) {
        setLogs(Array.isArray(data.operation_log) ? data.operation_log : []);
      }
    } catch (error) {
      console.error("operation log error", error);
    }
  }

  async function refreshPendingAction() {
    if (!currentUser) {
      return;
    }

    try {
      const response = await fetch(`${apiBaseUrl}/pending-action`, {
        credentials: "include",
      });
      if (response.status === 401) {
        setCurrentUser(null);
        resetWorkspace();
        return;
      }

      const data = await response.json();
      if (data.enabled === false) {
        setApprovalsEnabled(false);
        setPendingAction(null);
        return;
      }
      setPendingAction(data.pending_action ?? null);
    } catch (error) {
      console.error("pending action error", error);
    }
  }

  async function refreshFeishuReadiness() {
    try {
      const response = await fetch(`${apiBaseUrl}/feishu/config-status`, {
        credentials: "include",
      });
      if (response.status === 401) {
        setCurrentUser(null);
        resetWorkspace();
        return;
      }
      if (response.ok) {
        setFeishuReadiness((await response.json()) as FeishuReadiness);
      }
    } catch (error) {
      console.error("feishu readiness error", error);
    }
  }

  async function refreshOrders() {
    try {
      const params = new URLSearchParams({
        page: String(orderPagination.page),
        page_size: String(orderPagination.page_size),
      });
      if (orderStatusFilter) {
        params.set("status", orderStatusFilter);
      }
      if (orderTypeFilter) {
        params.set("order_type", orderTypeFilter);
      }
      const response = await fetch(`${apiBaseUrl}/orders?${params.toString()}`, {
        credentials: "include",
      });
      if (response.status === 401) {
        setCurrentUser(null);
        resetWorkspace();
        return;
      }
      const data = await response.json();
      if (!data.error) {
        setOrders(Array.isArray(data.orders) ? data.orders : []);
        if (data.pagination) {
          setOrderPagination(data.pagination as OrderPagination);
        }
      }
    } catch (error) {
      console.error("orders error", error);
    }
  }

  async function openOrderDetail(orderId: number) {
    setIsOrderBusy(true);
    try {
      const response = await fetch(`${apiBaseUrl}/orders/${orderId}`, {
        credentials: "include",
      });
      const data = await response.json();
      if (!response.ok) {
        setErrorText(data.detail ?? "订单详情读取失败。");
        return;
      }
      const order = data.order as OrderDetail;
      setSelectedOrder(order);
      const detail = order.detail ?? {};
      setEditStartDate(String(detail.checkin_date ?? detail.pickup_at ?? detail.visit_date ?? "").slice(0, 10));
      setEditEndDate(String(detail.checkout_date ?? detail.return_at ?? "").slice(0, 10));
      setEditFlightId("");
      setEditParticipantCount(String(detail.participant_count ?? 1));
    } catch (error) {
      console.error("order detail error", error);
      setErrorText("订单详情读取失败，请检查后端服务。");
    } finally {
      setIsOrderBusy(false);
    }
  }

  async function requestOrderAction(kind: "cancel" | "update" | "retry") {
    if (!selectedOrder || isOrderBusy) {
      return;
    }
    setIsOrderBusy(true);
    try {
      let body: Record<string, unknown> | undefined;
      if (kind === "update") {
        body = selectedOrder.order_type === "flight"
          ? { new_flight_id: Number(editFlightId) }
          : selectedOrder.order_type === "trip"
            ? { start_date: editStartDate, participant_count: Number(editParticipantCount) }
            : { start_date: editStartDate, end_date: editEndDate };
      }
      const response = await fetch(
        `${apiBaseUrl}/orders/${selectedOrder.id}/${kind}-request`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: "include",
          body: body ? JSON.stringify(body) : undefined,
        },
      );
      const data = await response.json();
      if (!response.ok) {
        setErrorText(data.detail ?? "订单操作提交失败。");
        return;
      }
      setPendingAction(data.pending_action ?? null);
      setSelectedOrder(null);
      setErrorText("");
    } catch (error) {
      console.error("order action error", error);
      setErrorText("订单操作提交失败，请检查后端服务。");
    } finally {
      setIsOrderBusy(false);
    }
  }

  async function submitAuth(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (isAuthenticating) {
      return;
    }

    setIsAuthenticating(true);
    setErrorText("");

    try {
      const endpoint = authMode === "login" ? "/auth/login" : "/auth/register";
      const payload =
        authMode === "login"
          ? { username, password }
          : {
              username,
              password,
              passenger_id: passengerId.trim() || undefined,
            };

      const response = await fetch(`${apiBaseUrl}${endpoint}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        credentials: "include",
        body: JSON.stringify(payload),
      });
      const data = await response.json();

      if (!response.ok) {
        setErrorText(data.error ?? "认证失败，请稍后重试。");
        return;
      }

      if (data.user) {
        setCurrentUser(data.user as AuthUser);
      }
      setPassword("");
      await Promise.all([
        bootstrapSession(),
        refreshOperationLog(),
        refreshPendingAction(),
        refreshOrders(),
        refreshFeishuReadiness(),
      ]);
    } catch (error) {
      console.error("submit auth error", error);
      setErrorText("认证请求发送失败，请检查后端服务状态。");
    } finally {
      setIsAuthenticating(false);
    }
  }

  async function logout() {
    try {
      await fetch(`${apiBaseUrl}/auth/logout`, {
        method: "POST",
        credentials: "include",
      });
    } catch (error) {
      console.error("logout error", error);
    } finally {
      setCurrentUser(null);
      resetWorkspace();
      setPassword("");
      setErrorText("");
    }
  }

  async function sendMessage(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const nextMessage = draft.trim();
    if (!currentUser || !nextMessage || isSending) {
      return;
    }

    setMessages((current) => [
      ...current,
      { role: "user", content: nextMessage },
    ]);
    setDraft("");
    setIsSending(true);
    setErrorText("");

    try {
      const response = await fetch(`${apiBaseUrl}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        credentials: "include",
        body: JSON.stringify({ message: nextMessage }),
      });

      if (response.status === 401) {
        setCurrentUser(null);
        resetWorkspace();
        setErrorText("登录状态已失效，请重新登录。");
        return;
      }

      const data = await response.json();
      const assistantMessage =
        data.response ?? data.error ?? "请求失败，请稍后再试。";

      setMessages((current) => [
        ...current,
        { role: "assistant", content: assistantMessage },
      ]);
      setChatHistory((current) => [
        ...current,
        {
          timestamp: new Date().toISOString(),
          user_message: nextMessage,
          ai_response: assistantMessage,
        },
      ]);

      await Promise.all([refreshOperationLog(), refreshPendingAction()]);
    } catch (error) {
      console.error("send message error", error);
      const fallbackMessage = "当前无法连接后端，请确认接口服务已经启动。";
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: fallbackMessage,
        },
      ]);
      setChatHistory((current) => [
        ...current,
        {
          timestamp: new Date().toISOString(),
          user_message: nextMessage,
          ai_response: fallbackMessage,
        },
      ]);
    } finally {
      setIsSending(false);
    }
  }

  async function submitDecision(decision: "approve" | "reject") {
    if (isDeciding) {
      return;
    }
    const endpoint =
      decision === "approve" ? "/approve-action" : "/reject-action";

    setIsDeciding(true);
    try {
      const response = await fetch(`${apiBaseUrl}${endpoint}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        credentials: "include",
        body: JSON.stringify({ decision }),
      });

      if (response.status === 401) {
        setCurrentUser(null);
        resetWorkspace();
        setErrorText("登录状态已失效，请重新登录。");
        return;
      }

      const data = await response.json();
      const assistantMessage =
        data.response ?? data.error ?? "审批处理失败，请稍后重试。";

      setMessages((current) => [
        ...current,
        { role: "assistant", content: assistantMessage },
      ]);
      setPendingAction(null);
      await Promise.all([refreshOperationLog(), refreshOrders()]);
    } catch (error) {
      console.error("submit decision error", error);
      setErrorText("审批请求发送失败，请检查后端服务状态。");
    } finally {
      setIsDeciding(false);
    }
  }

  if (!currentUser) {
    return (
      <main className="auth-shell">
        <section className="auth-hero">
          <p className="stage-kicker">Travel Identity</p>
          <h1>先登录，再把订单真正绑定到用户</h1>
          <p>
            登录后，当前会话会自动映射到你的乘机人 ID，航班查询、酒店预订和租车记录就能和当前用户关联起来。
          </p>
          <div className="hero-pills">
            <span>用户识别</span>
            <span>订单归属</span>
            <span>连续会话</span>
          </div>
        </section>

        <section className="auth-card">
          <div className="auth-tabs">
            <button
              type="button"
              className={authMode === "login" ? "active" : ""}
              onClick={() => setAuthMode("login")}
            >
              登录
            </button>
            <button
              type="button"
              className={authMode === "register" ? "active" : ""}
              onClick={() => setAuthMode("register")}
            >
              注册
            </button>
          </div>

          <form className="auth-form" onSubmit={submitAuth}>
            <label>
              用户名
              <input
                value={username}
                onChange={(event) => setUsername(event.target.value)}
                placeholder="请输入用户名"
                disabled={isAuthenticating}
              />
            </label>

            <label>
              密码
              <input
                type="password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                placeholder="至少 6 位密码"
                disabled={isAuthenticating}
              />
            </label>

            {authMode === "register" ? (
              <label>
                乘机人 ID（可选）
                <input
                  value={passengerId}
                  onChange={(event) => setPassengerId(event.target.value)}
                  placeholder="如需绑定已有机票，可填写现有 passenger_id"
                  disabled={isAuthenticating}
                />
              </label>
            ) : null}

            <p className="auth-hint">
              {authMode === "register"
                ? "不填写乘机人 ID 也可以注册，系统会自动生成一个新的用户标识。"
                : "登录后将恢复你的专属会话和对应订单记录。"}
            </p>

            <button type="submit" disabled={isAuthenticating}>
              {isAuthenticating
                ? "提交中..."
                : authMode === "login"
                  ? "立即登录"
                  : "创建账号"}
            </button>

            <p className="auth-error">
              {errorText || "登录后即可开始按当前用户身份查询和预订。"}
            </p>
          </form>
        </section>
      </main>
    );
  }

  return (
    <main className="dashboard-shell">
      <aside className="workspace-sidebar">
        <div className="sidebar-head">
          <p className="sidebar-kicker">Workspace</p>
          <h1>工作目录</h1>
          <p className="sidebar-copy">
            管理当前会话、查看调用工具、跟踪上下文状态。
          </p>
        </div>

        <section className="sidebar-section">
          <div className="section-heading">
            <h2>用户信息</h2>
            <button type="button" className="sidebar-action" onClick={() => void logout()}>
              退出
            </button>
          </div>
          <div className="context-card">
            <p>当前用户</p>
            <code>{currentUser.username}</code>
            <p>乘机人 ID</p>
            <code>{currentUser.passenger_id}</code>
          </div>
        </section>

        <section className="sidebar-section">
          <div className="section-heading">
            <h2>会话概览</h2>
            <span>{modeLabel}</span>
          </div>
          <div className="metric-grid">
            <article className="metric-card">
              <strong>{chatHistory.length}</strong>
              <span>历史轮次</span>
            </article>
            <article className="metric-card">
              <strong>{recentToolLogs.length}</strong>
              <span>最近工具</span>
            </article>
            <article className="metric-card">
              <strong>{pendingAction ? pendingAction.tool_calls.length : 0}</strong>
              <span>待审批</span>
            </article>
            <article className="metric-card">
              <strong>{orders.length}</strong>
              <span>我的订单</span>
            </article>
          </div>
          <div className="context-card">
            <p>Session ID</p>
            <code>{sessionId || "初始化中..."}</code>
            <p>当前状态</p>
            <span>{statusText}</span>
            <p>飞书审批</p>
            <span className={`feishu-summary ${feishuReadiness?.enabled ? "ready" : "pending"}`}>
              {feishuStatusText}
            </span>
          </div>
        </section>

        <section className="sidebar-section">
          <div className="section-heading">
            <h2>个人订单</h2>
            <button
              type="button"
              className="sidebar-action"
              onClick={() => void refreshOrders()}
            >
              刷新
            </button>
          </div>
          <div className="order-filters">
            <select
              aria-label="订单类型"
              value={orderTypeFilter}
              onChange={(event) => {
                setOrderTypeFilter(event.target.value);
                setOrderPagination((current) => ({ ...current, page: 1 }));
              }}
            >
              <option value="">全部类型</option>
              <option value="flight">航班</option>
              <option value="hotel">酒店</option>
              <option value="car">租车</option>
              <option value="trip">行程</option>
            </select>
            <select
              aria-label="订单状态"
              value={orderStatusFilter}
              onChange={(event) => {
                setOrderStatusFilter(event.target.value);
                setOrderPagination((current) => ({ ...current, page: 1 }));
              }}
            >
              <option value="">全部状态</option>
              <option value="confirmed">已确认</option>
              <option value="cancelled">已取消</option>
              <option value="processing">处理中</option>
              <option value="failed">失败</option>
            </select>
          </div>
          <div className="order-list">
            {orders.length === 0 ? (
              <div className="sidebar-empty">审批通过的订单会显示在这里。</div>
            ) : (
              orders.map((order) => (
                <button
                  type="button"
                  key={order.id}
                  className="order-card"
                  onClick={() => void openOrderDetail(order.id)}
                  disabled={isOrderBusy}
                >
                  <div className="order-topline">
                    <strong>{order.order_no}</strong>
                    <span>{order.status}</span>
                  </div>
                  <p>
                    {order.order_type.toUpperCase()} · {order.currency}{" "}
                    {(order.total_amount_minor / 100).toFixed(2)}
                  </p>
                  <small>
                    确认号：{order.supplier_confirmation_no ?? "处理中"}
                  </small>
                </button>
              ))
            )}
          </div>
          {orderPagination.total > orderPagination.page_size ? (
            <div className="order-pagination">
              <button
                type="button"
                disabled={orderPagination.page <= 1}
                onClick={() => setOrderPagination((current) => ({ ...current, page: current.page - 1 }))}
              >
                上一页
              </button>
              <span>
                {orderPagination.page} / {Math.ceil(orderPagination.total / orderPagination.page_size)}
              </span>
              <button
                type="button"
                disabled={orderPagination.page * orderPagination.page_size >= orderPagination.total}
                onClick={() => setOrderPagination((current) => ({ ...current, page: current.page + 1 }))}
              >
                下一页
              </button>
            </div>
          ) : null}
        </section>

        <section className="sidebar-section">
          <div className="section-heading">
            <h2>当前上下文</h2>
          </div>
          <div className="context-card">
            <p>后端地址</p>
            <code>{apiBaseUrl}</code>
            <p>最近助手回复</p>
            <span>{latestAssistantMessage || "等待对话开始..."}</span>
          </div>
        </section>

        <section className="sidebar-section">
          <div className="section-heading">
            <h2>会话历史记录</h2>
            <span>最近 {recentHistory.length} 条</span>
          </div>
          <div className="history-list">
            {recentHistory.length === 0 ? (
              <div className="sidebar-empty">还没有会话历史。</div>
            ) : (
              recentHistory.map((entry, index) => (
                <article
                  key={`${entry.timestamp ?? "history"}-${index}`}
                  className="history-card"
                >
                  <div className="history-topline">
                    <strong>用户提问</strong>
                    <span>{formatTimeLabel(entry.timestamp)}</span>
                  </div>
                  <p>{entry.user_message}</p>
                  <div className="history-divider" />
                  <small>{entry.ai_response}</small>
                </article>
              ))
            )}
          </div>
        </section>

        <section className="sidebar-section">
          <div className="section-heading">
            <h2>调用工具</h2>
            <button
              type="button"
              className="sidebar-action"
              onClick={() => void refreshOperationLog()}
            >
              刷新
            </button>
          </div>
          <div className="tool-list">
            {recentToolLogs.length === 0 ? (
              <div className="sidebar-empty">还没有工具调用记录。</div>
            ) : (
              recentToolLogs.map((entry, index) => (
                <article
                  key={`${entry.timestamp}-${index}`}
                  className={`tool-card ${entry.type}`}
                >
                  <div className="tool-topline">
                    <strong>{entry.title}</strong>
                    <span>{formatTimeLabel(entry.timestamp)}</span>
                  </div>
                  <p>{entry.content}</p>
                </article>
              ))
            )}
          </div>
        </section>
      </aside>

      <section className="conversation-stage">
        <header className="stage-hero">
          <div>
            <p className="stage-kicker">Agent Workspace</p>
            <h2>深圳出行智能客服</h2>
            <p>
              当前已按用户身份登录，后续机票、酒店和租车操作都会优先与当前乘机人 ID 关联。
            </p>
          </div>
          <div className="hero-pills">
            <span>Agent 调度</span>
            <span>混合检索</span>
            <span>结构化日志</span>
          </div>
        </header>

        <div className="conversation-card">
          <div className="conversation-header">
            <div>
              <p className="stage-kicker">Conversation</p>
              <h3>对话列表</h3>
            </div>
            <span className="conversation-badge">{modeLabel}</span>
          </div>

          <div className="message-feed">
            {messages.length === 0 ? (
              <div className="message-empty">
                <p>当前还没有会话内容。</p>
                <p>现在可以直接说：“帮我查一下我名下的航班信息”。</p>
              </div>
            ) : (
              messages.map((message, index) => (
                <article
                  key={`${message.role}-${index}`}
                  className={`message-row ${message.role}`}
                >
                  <span className="message-tag">
                    {message.role === "user" ? "你" : "助手"}
                  </span>
                  <div className="message-bubble">
                    <p>{message.content}</p>
                  </div>
                </article>
              ))
            )}
          </div>

          <form className="composer-panel" onSubmit={sendMessage}>
            <textarea
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              placeholder="输入你的问题，例如：帮我看看我当前账号关联的航班信息"
              rows={4}
              disabled={isBooting || isSending}
            />
            <div className="composer-footer">
              <div className="composer-hint">
                {errorText || "支持连续多轮追问，会自动保留当前会话上下文。"}
              </div>
              <button type="submit" disabled={isBooting || isSending}>
                {isSending ? "发送中..." : "发送"}
              </button>
            </div>
          </form>
        </div>
      </section>

      {selectedOrder ? (
        <div className="modal-backdrop">
          <div className="modal-card order-detail-modal">
            <div className="detail-heading">
              <div>
                <p className="modal-kicker">Order Detail</p>
                <h3>{selectedOrder.order_no}</h3>
              </div>
              <button type="button" className="close-button" onClick={() => setSelectedOrder(null)}>
                关闭
              </button>
            </div>
            <div className="detail-summary">
              <span>{selectedOrder.order_type.toUpperCase()}</span>
              <strong>{selectedOrder.currency} {(selectedOrder.total_amount_minor / 100).toFixed(2)}</strong>
              <span>{selectedOrder.status}</span>
            </div>
            <dl className="detail-grid">
              <div><dt>供应商</dt><dd>{selectedOrder.supplier_name}</dd></div>
              <div><dt>确认号</dt><dd>{selectedOrder.supplier_confirmation_no ?? "处理中"}</dd></div>
              {getOrderDetailRows(selectedOrder).map(([label, value]) => (
                <div key={label}><dt>{label}</dt><dd>{value}</dd></div>
              ))}
            </dl>
            {selectedOrder.supplier_attempts.length > 0 ? (
              <div className="supplier-attempts">
                <h4>供应商处理记录</h4>
                {selectedOrder.supplier_attempts.map((attempt, index) => (
                  <p key={`${String(attempt.operation)}-${index}`}>
                    {String(attempt.operation).toUpperCase()} · {String(attempt.status)}
                    {attempt.error_message ? ` · ${String(attempt.error_message)}` : ""}
                  </p>
                ))}
              </div>
            ) : null}
            {selectedOrder.status === "confirmed" ? (
              <div className="order-edit-panel">
                <h4>修改订单</h4>
                {selectedOrder.order_type === "flight" ? (
                  <label>
                    新航班 ID
                    <input
                      type="number"
                      min="1"
                      value={editFlightId}
                      onChange={(event) => setEditFlightId(event.target.value)}
                    />
                  </label>
                ) : (
                  <label>
                    {selectedOrder.order_type === "trip" ? "出行日期" : "开始日期"}
                    <input
                      type="date"
                      value={editStartDate}
                      onChange={(event) => setEditStartDate(event.target.value)}
                    />
                  </label>
                )}
                {selectedOrder.order_type === "hotel" || selectedOrder.order_type === "car" ? (
                  <label>
                    结束日期
                    <input
                      type="date"
                      value={editEndDate}
                      onChange={(event) => setEditEndDate(event.target.value)}
                    />
                  </label>
                ) : null}
                {selectedOrder.order_type === "trip" ? (
                  <label>
                    参与人数
                    <input
                      type="number"
                      min="1"
                      value={editParticipantCount}
                      onChange={(event) => setEditParticipantCount(event.target.value)}
                    />
                  </label>
                ) : null}
                <div className="modal-actions">
                  <button
                    type="button"
                    className="approve-button"
                    disabled={isOrderBusy}
                    onClick={() => void requestOrderAction("update")}
                  >
                    提交修改审批
                  </button>
                  <button
                    type="button"
                    className="reject-button"
                    disabled={isOrderBusy}
                    onClick={() => void requestOrderAction("cancel")}
                  >
                    提交取消审批
                  </button>
                </div>
              </div>
            ) : null}
            {selectedOrder.status === "failed" ? (
              <div className="modal-actions">
                <button
                  type="button"
                  className="approve-button"
                  disabled={isOrderBusy}
                  onClick={() => void requestOrderAction("retry")}
                >
                  提交重试审批
                </button>
              </div>
            ) : null}
          </div>
        </div>
      ) : null}

      {pendingAction ? (
        <div className="modal-backdrop">
          <div className="modal-card">
            <p className="modal-kicker">Approval Required</p>
            <h3>检测到待审批动作</h3>
            <p className="modal-copy">
              批准后才会创建订单。当前仍可在本页面确认。
            </p>
            <div className={`feishu-approval-status ${feishuReadiness?.enabled ? "ready" : "pending"}`}>
              <strong>飞书审批</strong>
              <span>{feishuStatusText}</span>
            </div>
            <div className="approval-summary">
              <strong>审批内容</strong>
              <p>
                {pendingAction.approval_summary ?? "审批详情暂不可用，请刷新后重试。"}
              </p>
            </div>
            <div className="modal-actions">
              <button
                type="button"
                className="approve-button"
                onClick={() => void submitDecision("approve")}
                disabled={isDeciding}
              >
                {isDeciding ? "处理中..." : "批准"}
              </button>
              <button
                type="button"
                className="reject-button"
                onClick={() => void submitDecision("reject")}
                disabled={isDeciding}
              >
                拒绝
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </main>
  );
}
