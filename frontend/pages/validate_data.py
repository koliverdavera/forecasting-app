import reflex as rx

from frontend.state import State


def _col_item(col: str) -> rx.Component:
    return rx.list_item(rx.code(col))


def _error_item(err: str) -> rx.Component:
    return rx.box(
        rx.text(err, color="red"),
        padding="0.4em 0.8em",
        border_left="3px solid red",
        margin_bottom="0.5em",
    )


def validate_data() -> rx.Component:
    State.check_authentication()

    valid_summary = rx.vstack(
        rx.markdown("### Загружены следующие данные:"),
        rx.ordered_list(
            rx.list_item(
                rx.hstack(
                    rx.text("Дата первого наблюдения — "),
                    rx.code(State.first_date),
                )
            ),
            rx.list_item(
                rx.hstack(
                    rx.text("Частота временного ряда — "),
                    rx.code(State.data_freq),
                )
            ),
            rx.list_item("Пропуски в обучающей выборке отсутствуют"),
            rx.list_item(
                rx.vstack(
                    rx.text("Драйверы (будут предсказаны моделью):"),
                    rx.unordered_list(rx.foreach(State.driver_cols, _col_item)),
                    spacing="1",
                    align_items="start",
                )
            ),
            rx.list_item(
                rx.vstack(
                    rx.text("Сценарные ряды (входные признаки для модели):"),
                    rx.unordered_list(rx.foreach(State.input_cols, _col_item)),
                    spacing="1",
                    align_items="start",
                )
            ),
            rx.cond(
                State.has_total,
                rx.list_item(
                    rx.vstack(
                        rx.text("Колонка total вычисляется по формуле:"),
                        rx.code(State.total_formula_display),
                        spacing="1",
                        align_items="start",
                    )
                ),
                rx.fragment(),
            ),
            rx.list_item(
                rx.hstack(
                    rx.text("Период прогноза: с "),
                    rx.code(State.date),
                    rx.text(" по "),
                    rx.code(State.forecast_end),
                )
            ),
        ),
        spacing="3",
        align_items="start",
    )

    error_summary = rx.vstack(
        rx.box(
            rx.text(
                "Данные не прошли валидацию. Исправьте файл и загрузите снова.",
                color="red",
                weight="bold",
            ),
            padding="1em",
            border="1px solid red",
            border_radius="8px",
            width="100%",
            margin_bottom="1em",
        ),
        rx.foreach(State.validation_errors, _error_item),
        spacing="2",
        align_items="start",
        width="100%",
    )

    buttons = rx.vstack(
        rx.button(
            "Верно, запустить обучение",
            on_click=[
                rx.redirect('/graph'),
                State.plot_finals()
            ],
            width="100%",
            disabled=~State.is_valid,
        ),
        rx.button(
            "Неверно, удалить загруженные данные",
            on_click=[
                rx.clear_selected_files("upload1"),
                rx.redirect('/'),
                State.clean()
            ],
            width="100%",
        ),
        width="50vw",
        align="center",
        justify="between",
    )

    main_grid = rx.vstack(
        rx.cond(State.is_valid, valid_summary, error_summary),
        rx.divider(width="100%", margin="20px auto", border_color="#dcdcdc"),
        rx.data_table(
            data=State.preview_document,
            pagination=True,
            search=False,
            sort=False,
            resizable=True,
        ),
        rx.cond(
            State.is_valid,
            rx.markdown("**Все верно, переходим к обучению моделей?**"),
            rx.fragment(),
        ),
        buttons,
        padding="5em",
        width="80vw",
        align="center",
        justify="between",
    )

    return main_grid
