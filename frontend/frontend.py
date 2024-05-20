import reflex as rx

from frontend.pages.auth import authorise
from frontend.pages.graph import graph
from frontend.pages.upload import upload
from frontend.pages.validate_data import validate_data

app = rx.App(
    theme=rx.theme(
        appearance="light", has_background=True, radius="large", accent_color="teal"
    )
)
app.add_page(authorise, route="/")
app.add_page(upload, route="/upload")
app.add_page(validate_data, route="/validate")
app.add_page(graph, route="/graph")
