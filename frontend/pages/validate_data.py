import numpy as np
import pandas as pd
import reflex as rx

from frontend.state import State


def validate_data() -> rx.Component:
    State.check_authentication()
    data = (
        pd.read_excel('data/test_data_input.xlsx', index_col=0)
        .head(3).reset_index()
    )
    data.iloc[:, 1:] = np.round(data.iloc[:, 1:], 2)

    confirm_data = rx.markdown(
        """ 
        ### Загружены следующие данные:

        1. Дата первого наблюдения –  **2016-01-01**
        2. Пропуски в обучающей выборке отсутствуют
        3. Загружено 3 ряда – драйвера, которые будут предсказаны с помощью модели
            - `driver_1_transactions`
            - `driver_2_volume`
            - `driver_3_transactions`,
        4. Загружено 3 сценарных ряда, которые являются входными признаками для модели
            - `input_1_mean_commission`
            - `input_2_mean_margin`, 
            - `input_3_cost_of_payment`, 
        5. Загружена колонка `total` с общим значением метрики, вычисляемая по следующей формуле:  

            `driver_1_transactions * input_1_mean_commission + driver_2_volume * input_2_mean_margin + driver_3_transactions * input_3_cost_of_payment`

        4. Модель обучится предсказывать значения с 2024-01-01 по 2024-03-31

        Начало загруженного файла:

        """
    )

    data_table = rx.data_table(
        data=data,
        pagination=True,
        search=False,
        sort=False,
        resizable=True
    )

    buttons = rx.vstack(
        rx.button(
            "Верно, запустить обучение",
            on_click=[
                rx.redirect('/graph'),
                State.plot_finals()
            ],
            width="100%",
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
        jusitfy="between"
    )

    main_grid = rx.vstack(
        confirm_data,
        rx.divider(width="100%", margin="20px auto", border_color="#dcdcdc"),
        data_table,
        rx.markdown(
            """
            **Все верно, переходим к обучению моделей?**
            """
        ),
        buttons,
        padding="5em",
        width="80vw",
        align="center",
        jusitfy="between"
    )

    return main_grid
