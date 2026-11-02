# IoT 控制台

基于 Flask 的物联网温湿度监控控制台（讲义 6 示例项目）。

## 功能

- 监控环境温度
- 监控环境湿度
- 设置目标温度

## 项目结构

```
.
├── app.py              # Flask 主程序
├── templates/
│   └── index.html      # 首页模板（Jinja2）
├── requirements.txt    # 依赖清单
└── README.md
```

## 运行

```bash
pip install -r requirements.txt
python app.py
```

启动后浏览器访问 http://localhost:5000

监听 `0.0.0.0`，局域网内其他设备也能访问；端口可通过环境变量 `PORT` 覆盖（默认 5000）。

## 路由

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 首页，展示当前温度、湿度、目标温度 |
| POST | `/set-temperature` | 接收表单字段 `target_temperature`，更新全局目标温度后重定向回 `/` |

## 说明

温湿度数据由 `get_latest_data()` 返回，目前是硬编码的模拟值（25.0 ℃ / 60.0 %），
接入真实传感器时替换该函数的实现即可。

目标温度存在全局变量里，进程重启即清空，未做持久化。
