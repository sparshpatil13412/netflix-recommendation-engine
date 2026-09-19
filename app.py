from flask import Flask, render_template, request
from recommendation import recommend

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            title = request.form["title"]
            title_type = request.form["title_type"]
            genres = request.form["genres"]
            rating = request.form["rating"]
            release_year = request.form["release_year"]
            duration = request.form["duration"]

            recommendations = recommend(
                title,
                title_type,
                genres,
                rating,
                release_year,
                duration
            )

            return render_template(
                "index.html",
                recommendations=recommendations.to_dict("records"),
                submitted=True
            )

        except ValueError as e:
            return render_template(
                "index.html",
                error=str(e)
            )
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)