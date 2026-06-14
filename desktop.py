"""DMORE 智能脸谱 - 桌面应用入口。
内置启动本地服务 + 打开原生窗口（无浏览器外壳）。
"""
import os, sys, time, socket, threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PORT = 8800
URL = f"http://127.0.0.1:{PORT}/"


def port_busy(p):
    s = socket.socket()
    s.settimeout(0.3)
    try:
        s.connect(("127.0.0.1", p)); s.close(); return True
    except Exception:
        return False


def start_server():
    import uvicorn
    from server.main import app
    config = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning")
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None   # 非主线程不能装信号
    server.run()


def main():
    if not port_busy(PORT):
        threading.Thread(target=start_server, daemon=True).start()
        for _ in range(80):
            if port_busy(PORT):
                break
            time.sleep(0.25)
    if os.environ.get("DMORE_NO_WINDOW") == "1":
        print("server up at", URL, flush=True)
        # 测试模式：保持运行
        while True:
            time.sleep(1)
    import webview
    webview.create_window("DMORE 智能脸谱", URL, width=1480, height=920, min_size=(1100, 700))
    webview.start()


if __name__ == "__main__":
    main()
