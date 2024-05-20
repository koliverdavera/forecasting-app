import reflex as rx

from frontend.state import State


def authorise() -> rx.Component:
    login_form = rx.box(
        rx.vstack(
            rx.form(
                rx.fragment(
                    rx.heading(
                        "Войдите в свой аккаунт",
                        size="7",
                        margin_bottom="2rem",
                    ),
                    rx.text(
                        "Логин",
                        margin_top="2px",
                        margin_bottom="4px",
                    ),
                    rx.input(
                        placeholder="username",
                        id="username",
                        justify_content="center",
                    ),
                    rx.text(
                        "Пароль",
                        margin_top="2px",
                        margin_bottom="4px",
                    ),
                    rx.input(
                        placeholder="password",
                        id="password",
                        justify_content="center",
                        type="password",
                    ),
                    rx.box(
                        rx.button(
                            "Войти",
                            type="submit",
                            width="100%",
                        ),
                        padding_top="14px",
                    ),
                ),
                on_submit=State.login_submit,
            ),
            align_items="center",
        ),
        padding="8rem 10rem",
        margin_top="10vh",
        margin_x="auto",
    )
    return login_form
