from flask import Flask
# from app.routes.user_routes import token_bp, stock_bp, UserStock_bp
from app.routes.user_routes import  UserStock_bp, data_resampling_bp, indicators_bp,nested_array_bp


def create_app():
    app = Flask(__name__)
    # app.register_blueprint(token_bp, url_prefix='/api')
    # app.register_blueprint(stock_bp, url_prefix='/api')
    app.register_blueprint(UserStock_bp, url_prefix="/api")
    app.register_blueprint(data_resampling_bp, url_prefix="/api")
    app.register_blueprint(indicators_bp, url_prefix="/api")
    app.register_blueprint(nested_array_bp, url_prefix="/api")

    return app
