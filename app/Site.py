from flask import Flask, render_template, request, redirect, url_for, g
import mysql.connector as db
from database import insert_into_users, validate_user, insert_into_post, get_post_data
from model import Model, add_data
from images import save_image_as
import sys
import os


app = Flask(__name__)

my_db = db.connect(
        host="localhost",
        user="root",
        password="root",
        database="websitedata"
    )


@app.route("/", methods=["GET"])
def home():
    return render_template("home.html", request=request)

@app.route("/Register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        email = request.form.get("email")
        phone = request.form.get("phone")
        address = request.form.get("address")

        data = (username, password, email, phone, address)

        insert_into_users(data=data, my_db=my_db)

        return redirect(url_for('login'))

    return render_template("register.html", request=request)

@app.route("/login", methods=["GET", "POST"])
def login():
    flash_message="False"
    if request.method == "POST":
        flash_message = "True"

        username = request.form.get("username")
        password = request.form.get("password")

        credentials = (username, password)

        user_is_valid = validate_user(credentials, my_db)
        if user_is_valid:
            return redirect(url_for('send_post', id=user_is_valid, name=username))
        
    return render_template("signin.html",flash_message = flash_message)

@app.route("/sendPost/<id>/<name>", methods=["GET", "POST"])
def send_post(id,name):
    
    if request.method == "POST":
    
        message = request.form.get("message")
        messagetype = request.form.get("messagetype")
        add_data(message,messagetype)


    return render_template("sendpost.html", id=id, name=name)


@app.route("/runalgorithm/<id>/<name>", methods=["GET", "POST"])
def run_algorithm(id, name):
    if request.method == "POST":
        
        algorithm = request.form.get("algorithm")
        return redirect(url_for('see_algorithm_data', id=id, algorithm=algorithm, name=name))
        
        
    return render_template("runalgorithm.html", id=id, name=name)

@app.route("/algorithmdata/<id>/<name>", methods=["GET", "POST"])
def see_algorithm_data(id, name):
    
    algorithm = request.args.get("algorithm")

    mymodel = Model(algorithm)
    mymodel.train_model()
    
    accuracy = mymodel.get_accuracy_score()
    mymodel.get_confusion_matrix()
    
    if request.method == "POST":
    
        message = request.form.get('message')
        image = request.files['get_image'] 
        prediction = mymodel.predict_data(message)

        image_address= save_image_as(image.name)        
        image.save(os.path.join('.', 'static', image_address))

        sep = os.path.sep
        image_address = '/'.join(image_address.split(sep))

        insert_into_post(USER_ID=id, MESSAGE=message, PREDICTION=prediction, IMAGEPATH=image_address, my_db=my_db)

        return redirect(url_for('view_post', id=id, name=name))
    
    return render_template("algorithmdata.html",id=id, accuracy=accuracy, name=name)

@app.route('/viewroute/<id>/<name>')
def view_post(id, name):

    data = get_post_data(id=id, my_db=my_db)
    return render_template('viewpost.html', data=data, id=id, name=name)

if __name__ == "__main__":

    app.run(debug=True)
