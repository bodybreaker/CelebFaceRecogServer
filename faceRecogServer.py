from flask import Flask, render_template, request
import ssl

app = Flask(__name__)

@app.route('/')
def test():
    return 'test'

if __name__ == "__main__":
    app.debug = True
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS)
    ssl_context.load_cert_chain(certfile='/etc/letsencrypt/live/minwoo.org/fullchain.pem', keyfile='/etc/letsencrypt/live/minwoo.org/privkey.pem')
    app.run(host="0.0.0.0", port=8575, ssl_context=ssl_context)
