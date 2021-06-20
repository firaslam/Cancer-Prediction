from flask import Flask, request

app = Flask(__name__)

@app.route('/') #define home route
def home():
    return "Welcome to Home"

@app.route('/add')
def add():
    a = 10
    b = 20
    return str(a+b) # the return type must be a string, dict, tuple, Response instance, or WSGI callable

@app.route('/addParam', methods=['GET']) # GET means we explictly show the parameter values (such as a=10&b=20) in the URL
def add_param():
    a = request.args.get("a")
    b = request.args.get("b")
    return str(int(a)+int(b)) # put `http://127.0.0.1:5000/addParam?a=10&b=20`

@app.route('/addParamPost', methods=['POST']) # Open postman and paste `http://127.0.0.1:5000/addParamPost` fill in a,b in Body -> form-data
def add_param_post():
    """
    Use postman
    """
    a = request.form['a']
    b = request.form['b']
    return str(int(a)+int(b))


if __name__ == '__main__':
    app.run()
