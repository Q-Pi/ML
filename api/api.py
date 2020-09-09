from flask import *


app = Flask(__name__)


@app.route('/')
def index():
	return "<a href=\"" + url_for("book_hub") + "\" >Book</a>" + \
		"<br></br><a href=\"" + url_for("books") + "\" >Books</a>"


#@app.route('/api/books')
#def books():
#    return send_from_directory("data", "books.json", as_attachment=True)


@app.route('/api/books')
def books():
	return jsonify(json.load(open("data/books.json")))


@app.route('/api/book')
def book_hub():
	max = str(len(json.load(open("data/books.json"))))
	return "<form method=\"POST\">" + \
		"<input type=\"number\" name=\"book_id\" size=\"2\" value=\"0\" min=\"0\" max=\"" + max + "\"> (0-" + max + ")" + \
	    "<input type=\"submit\" value=\"Valider\">" + \
		"</form>"


@app.route('/api/book', methods=['POST'])
def book_hub_post():
	return redirect(url_for("book", id=request.form["book_id"]))


@app.route('/api/book/<id>')
def book(id):
	return jsonify(json.load(open("data/books.json"))[int(id)])


if __name__ == '__main__':
	app.run()

