# 开发者 API

## 速率限制

API 限流 60 次/分钟，按每个账号独立计算。超出限流后，服务端返回 429 状态码，响应头 `Retry-After` 会给出建议的等待秒数。

## 认证方式

在「设置 > 开发者」页面生成 API Token，请求时放在 `Authorization: Bearer <token>` 头中。Token 泄漏后可立即吊销重建。
