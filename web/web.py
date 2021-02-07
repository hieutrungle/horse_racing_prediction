import numpy
import pandas as pd
import os
from flask import Flask, render_template, url_for, flash, redirect

from forms import RegistrationForm, LoginForm

app = Flask(__name__)

app.config['SECRET_KEY'] = '6096dbdb10e65dbc14371167d86f79fc'


posts = [
    {
        'author': "Hieu Le",
        'title': 'Blog Post 1',
        'content': 'First post content',
        'date_posted': 'April 20, 2018'
    },
    {
        'author': "Linh Bui",
        'title': 'Blog Post 2',
        'content': 'Second post content',
        'date_posted': 'September 20, 2018'
    }
]

@app.route("/")
@app.route("/home/")
def home():
    return render_template('home.html', posts=posts)

@app.route("/about/")
def about():
    return render_template('about.html', title='about')

@app.route("/register", methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        flash(f'Account created for {form.username.data}!', 'success')
        return redirect(url_for('home'))

    return render_template('register.html', title='Register', form=form)

@app.route("/login", methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        if form.email.data == 'admin@blog.com' and form.password.data == 'password':
            flash('You have been logged in!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Login unsuccessful. Please check username and password', 'danger')
    return render_template('login.html', title='Login', form=form)


@app.route("/tables")
def show_tables():
    # Get data
    current_path = os.path.dirname(__file__)
    parent_path = os.path.dirname(current_path)
    data_path = parent_path + "/data/training.csv"
    data = pd.read_csv(data_path)
    data_head = data.iloc[:,1:].head()
    # data.set_index(['Name'], inplace=True)
    # data.index.name=None
    # females = data.loc[data.Gender=='f']
    # males = data.loc[data.Gender=='m']
    return render_template('table.html',tables=[data_head.to_html(classes='top3')], shape=data.shape,
                            titles = ['na', 'Top 3'])


if __name__ == '__main__':
    app.run(debug=True)