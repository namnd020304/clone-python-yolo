"""Gửi tin WhatsApp về điện thoại qua CallMeBot (miễn phí, không cần thư viện ngoài).

Lấy API key một lần: lưu số của CallMeBot vào danh bạ (số hiện tại xem ở
callmebot.com/blog/free-api-whatsapp-messages/), nhắn cho số đó "I allow callmebot to send me messages",
bot trả về apikey. Sau đó đặt 2 biến môi trường (mở terminal mới sau khi setx):

    setx WHATSAPP_PHONE +84912345678
    setx CALLMEBOT_APIKEY 1234567

Thử:  python -m yolo.notify "xin chào"
Chưa đặt biến môi trường thì send_whatsapp chỉ in cảnh báo, không làm hỏng chương trình.
"""
import os
import sys
import urllib.parse
import urllib.request


def send_whatsapp(text: str, timeout: float = 20.0) -> bool:
    """Trả về True nếu gửi được. Không bao giờ ném lỗi: dùng trong except của train.py, không được che lỗi gốc."""
    phone, key = os.environ.get("WHATSAPP_PHONE"), os.environ.get("CALLMEBOT_APIKEY")
    if not phone or not key:
        print("[notify] chưa đặt WHATSAPP_PHONE / CALLMEBOT_APIKEY, bỏ qua gửi WhatsApp", file=sys.stderr)
        return False
    query = urllib.parse.urlencode({"phone": phone, "text": text[:1500], "apikey": key})
    try:
        with urllib.request.urlopen(f"https://api.callmebot.com/whatsapp.php?{query}", timeout=timeout) as r:
            body = r.read().decode("utf-8", "replace")
        ok = r.status == 200 and "error" not in body.lower()
        if not ok:
            print(f"[notify] CallMeBot trả về: {body[:300]}", file=sys.stderr)
        return ok
    except Exception as e:  # mất mạng, timeout, ...
        print(f"[notify] gửi WhatsApp lỗi: {e}", file=sys.stderr)
        return False


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    msg = " ".join(sys.argv[1:]) or "Thử thông báo từ yolo.notify"
    print("Đã gửi" if send_whatsapp(msg) else "Gửi thất bại")
