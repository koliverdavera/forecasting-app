import reflex as rx

from frontend.state import FigureWithTitle, State


def plot_fig(fig: FigureWithTitle) -> rx.Component:
    return rx.vstack(
        rx.text(fig.title, width="100%", text_align="center", margin=0, weight="bold", size="5"),
        rx.plotly(data=fig.fig),
        spacing="0",
        padding=0,
        width="100%",
        align_items="center",
        justify_content="between"
    )


def download_forecasts() -> rx.download:
    with open("data/test_data_output.xlsx", "rb") as f:
        return rx.download(data=f.read(), filename='forecasts.xlsx')


def best_model_info() -> rx.Component:
    result = rx.vstack(
        rx.markdown(
            f"""
            <div style="text-align: center;">
                <h2 style="color: #2c3e50;">Лучшая модель – {State.best_model}</h2>
                <h3 style="color: #8e44ad;">Ошибки лучшей модели на целевой метрике:</h3>
                <h3 style="color: #8e44ad;">{State.best_score}</h3>
            </div>
            """,
            unsafe_allow_html=True
        ),
        padding="20px",
        border_radius="10px",
        box_shadow="0 4px 8px rgba(0, 0, 0, 0.1)",
        background_color="#ecf0f1",
        width="80%",
        margin="auto",
        align_items="center",
        justify_content="center",
        margin_bottom="20px"
    )
    return result


def graph() -> rx.Component:
    State.check_authentication()
    buttons = rx.hstack(
        rx.button(
            "В начало",
            on_click=[
                rx.redirect("/"),
                State.clean(),
            ],
            width="50%",
            height="50px",
            size="4",
            flex_grow=1
        ),
        rx.button("Скачать прогнозы", on_click=download_forecasts, width="50%", height="50px", size="4", flex_grow=1),
        width="100%",
        padding=10,
        justify_content="center",
        align_items="center"
    )

    divider = rx.divider(width="100%", margin="20px auto", border_color="#dcdcdc")

    return rx.vstack(
        rx.hstack(
            plot_fig(State.total),
            best_model_info(),
            width="100%",
            align_items="center",
            justify_content="center",
            spacing="0"
        ),
        divider,
        rx.hstack(
            rx.foreach(State.figs1[:3], plot_fig),
            width="100%",
            align_items="center",
            justify_content="center",
            spacing="0"
        ),
        rx.hstack(
            rx.foreach(State.figs2[:3], plot_fig),
            width="100%",
            align_items="center",
            justify_content="center",
            spacing="0"
        ),
        divider,
        buttons,
        width="100%",
        spacing="0",
        padding=20,
        align_items="center",
        justify_content="center",
        background_color="#f9f9f9"
    )
