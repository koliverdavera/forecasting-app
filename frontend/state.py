import re
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import reflex as rx
from openpyxl import load_workbook
from openpyxl.utils import column_index_from_string
from plotly.graph_objs import Figure


class FigureWithTitle(rx.Base):
    fig: Figure
    title: str


def authenticate_user(username, password):
    return True


class State(rx.State):
    """The app state."""

    figs1: List[FigureWithTitle] = []
    figs2: List[FigureWithTitle] = []
    total: FigureWithTitle = None
    loaded_document: pd.DataFrame = pd.DataFrame()
    predicted_document: pd.DataFrame = pd.DataFrame()
    date: str = ''
    best_model: str = 'Arima'
    best_score: str = 'SMAPE = 6.7%, MSE = 450.787'
    is_authenticated: bool = False
    print(datetime.now())

    async def handle_upload(self, files: list[rx.UploadFile]):
        """Handle the upload of file(s).

        Args:
            files: The uploaded files.
        """
        for file in files:
            upload_data = await file.read()
            outfile = rx.get_upload_dir() / file.filename

            with outfile.open("wb") as file_object:
                file_object.write(upload_data)

            self.parse_loaded_file(outfile)

    def parse_formula(self, formula_str: str = '', col_names: list = None):
        """Parse an Excel formula string and return a callable for row-wise application.

        Maps cell references like B2, C3 to DataFrame column names using col_names list,
        where col_names[0] corresponds to Excel column B (first data column after index).
        Falls back to a hardcoded formula if formula_str or col_names are not provided.
        """
        if not formula_str or not col_names:
            return lambda row: row.iloc[0] * row.iloc[1] + row.iloc[2] * row.iloc[3] + row.iloc[4] * row.iloc[5]

        # Remove leading '=' if present
        expr = formula_str.lstrip('=')

        # Replace Excel cell references (e.g. B2, C3) with DataFrame row accesses.
        # Excel column A (index=1) is the date index, so B (index=2) → col_names[0].
        def replace_ref(match):
            col_letter = match.group(1)
            col_idx = column_index_from_string(col_letter) - 2
            if 0 <= col_idx < len(col_names):
                return f'row["{col_names[col_idx]}"]'
            return match.group(0)

        expr = re.sub(r'([A-Z]+)\d+', replace_ref, expr)
        return eval(f'lambda row: {expr}')

    def predict(self, test=True):
        if test:
            self.predicted_document = pd.read_excel('data/test_data_output.xlsx', index_col=0)
            self.predicted_document.index = pd.to_datetime(self.predicted_document.index)
        else:
            # TODO add actual predictions
            self.predicted_document = pd.read_excel('data/test_data_output.xlsx', index_col=0)
            self.predicted_document.index = pd.to_datetime(self.predicted_document.index)
        self.predicted_document['total'] = self.predicted_document.apply(self.formula, axis=1)
        return self.predicted_document

    def plot_with_conf(self, col) -> Figure:
        df = self.predicted_document
        history = df[df.index < self.date][col]
        forecast = df[df.index >= self.date][col]

        # Estimate volatility from historical first differences.
        # For a random walk, forecast uncertainty grows as sigma * sqrt(h).
        sigma = history.diff().dropna().std()
        horizons = np.arange(1, len(forecast) + 1)
        ci_half = 1.96 * sigma * np.sqrt(horizons)

        fig = go.Figure()
        fig.add_traces(
            [
                go.Scatter(
                    x=history.index,
                    y=history.values,
                    mode="lines",
                    name='history'
                ),
                go.Scatter(
                    x=forecast.index,
                    y=forecast.values,
                    mode="lines",
                    name='forecast'
                ),
                go.Scatter(
                    x=forecast.index,
                    y=forecast.values + ci_half,
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    showlegend=False
                ),
                go.Scatter(
                    x=forecast.index,
                    y=forecast.values - ci_half,
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    name='95% confidence interval',
                    fill='tonexty',
                    fillcolor='rgba(255, 0, 0, 0.2)'
                )
            ]
        )
        return fig

    def plot_finals(self):
        self.predict()
        for cols in np.array(self.predicted_document.columns[:6]).reshape((3, 2)).tolist():
            figs = []
            for col in cols:
                fig = self.plot_with_conf(col=col)
                figs.append(FigureWithTitle(fig=fig, title=col))
            self.figs1.append(figs[0])
            self.figs2.append(figs[1])

        if 'total' in self.predicted_document.columns:
            self.total = FigureWithTitle(
                fig=self.plot_with_conf(col='total'),
                title='Total'
            )
            self.total.fig.write_image(f"assets/fig.png")

    def parse_loaded_file(self, outfile):
        df = pd.read_excel(outfile, index_col=0)
        df.index = pd.to_datetime(df.index)

        # TODO validation
        self.loaded_document = df
        self.date = str(df[df.isna().any(axis=1)].index.min())[:10]

        if 'total' in df.columns:
            # load_workbook reads formula strings by default (data_only=False)
            wb = load_workbook(outfile)
            sheet = wb.active
            # Read formula from row 2 of the last (total) column
            formula_str = sheet.cell(row=2, column=sheet.max_column).value
            # col_names maps Excel col B→[0], C→[1], … to DataFrame column names
            col_names = list(df.columns)
            self.formula = self.parse_formula(formula_str=formula_str, col_names=col_names)
        else:
            self.formula = self.parse_formula()

    def login_submit(self, form):
        # rx.redirect('/upload')

        self.login_error_message = ""
        username = form["username"]
        password = form["password"]
        is_authenticated = authenticate_user(username, password)

        if not is_authenticated:
            self.login_error_message = "He удалось войти в CIBAA LLM UI"
            return rx.set_value("password", "")

        # TODO validation
        self.is_authenticated = True
        return rx.redirect('/upload')

    def redirect(self) -> rx.Component:
        if not self.user.is_authenticated:
            return rx.box(rx.redirect("/"))
        return rx.box(rx.redirect("/upload"))

    def clean(self):
        self.loaded_document = pd.DataFrame()
        self.date = ''
        self.total = None
        self.figs1.clear()
        self.figs2.clear()

    def check_authentication(self):
        print(self.is_authenticated)
        return rx.cond(
            self.is_authenticated & True,
            rx.fragment(),
            self.redirect()
        )
