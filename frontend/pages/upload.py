import numpy as np
import pandas as pd
import reflex as rx

from frontend.state import State

color = "rgb(107,99,246)"


def upload() -> rx.Component:
    State.check_authentication()
    data = (
        pd.read_excel('data/test_data_input.xlsx', index_col=0)
        .head(5)
        .reset_index()
        .drop(columns=['driver_3_transactions', 'input_3_cost_of_payment'])
    )
    data.iloc[:, 1:] = np.round(data.iloc[:, 1:], 2)

    intro = rx.markdown(
        """ 
        ### Требования к структуре данных

        1. **Первая колонка** должна содержать даты.
        2. **Остальные колонки** должны содержать значения временных рядов. Значения могут быть целыми или нецелыми числами.
        3. Названия колонок (кроме дат) должны начинаться с подстрок:
           - `input_{номер_ряда}_{название ряда}` — для сценарных временных рядов.
           - `driver_{номер_ряда}_{название ряда}` — для рядов драйверов.
        4. **Опционально**: последняя колонка может содержать финальное значение, вычисляемое на основе формулы Excel, и должна начинаться с подстроки `total`.
        5. **Важное требование**: в файле не должно быть пропусков.

        """
    )

    data_table = rx.data_table(
        data=data,
        pagination=True,
        search=False,
        sort=False,
        resizable=True,
        width='50%'
    )

    upload_file = rx.upload(
        rx.vstack(
            rx.button(
                "Выбрать файл для загрузки", color=color, bg="white", border=f"1px solid {color}"
            ),
            rx.text("Перетяните файл в это окно или кликните для загрузки"),
            align="stretch",
            justify="between",
            height="100%",
            width="100%"
        ),
        id="upload1",
        border=f"1px dotted {color}",
        padding="5em",
        max_files=1,
        width="100%",
        height="100%",
        align="center",
        justify="between"
    )

    buttons = rx.vstack(
        rx.hstack(
            rx.foreach(rx.selected_files("upload1"), rx.text),
            width="20%",
            height="10%",
            margin=10,
            size="4",
            align="center",
            justify="between"
        ),
        rx.button(
            "Загрузить",
            on_click=[
                State.handle_upload(rx.upload_files(upload_id="upload1")),
                rx.redirect("/validate")
            ],
            width="100%",
        ),
        rx.button(
            "Удалить",
            on_click=rx.clear_selected_files("upload1"),
            width="100%",
        ),
        width="50vw",
        align="center",
        jusitfy="between"
    )

    main_grid = rx.vstack(
        intro,
        data_table,
        rx.vstack(
            upload_file,
            buttons,
            height="100%",
            align_items="center",
            justify_content="center",
            spacing="0"
        ),
        padding="5em",
        width="100vw",
        align="center",
        jusitfy="between"
    )

    return main_grid
