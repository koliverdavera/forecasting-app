import io
import sys
from copy import deepcopy
from datetime import datetime
from typing import Callable, List

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from etna.analysis import plot_backtest
from etna.datasets.tsdataset import TSDataset
from etna.ensembles import (
    VotingEnsemble, StackingEnsemble
)
from etna.metrics import SMAPE, MAPE, MSE
from etna.models import (
    CatBoostMultiSegmentModel,
    CatBoostPerSegmentModel,
    AutoARIMAModel,
    LinearMultiSegmentModel,
    LinearPerSegmentModel,
    ElasticMultiSegmentModel,
    ElasticPerSegmentModel,
    # ProphetModel,
    SklearnPerSegmentModel,
    SklearnMultiSegmentModel,
    SimpleExpSmoothingModel
)
from etna.pipeline import Pipeline, AutoRegressivePipeline
from etna.transforms import Transform
from etna.transforms.base import OneSegmentTransform, ReversiblePerSegmentWrapper
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor
)
from statsmodels.tsa.seasonal import STL
from tqdm import tqdm


class _OneSegmentSTLTrend(OneSegmentTransform):

    def __init__(self, in_column: str, robust: bool):
        self.in_column = in_column
        self.robust = robust

    def fit(self, df: pd.DataFrame) -> "_OneSegmentSTLTrend":
        target_column = df[self.in_column]
        self.init_values = target_column
        stl_model = STL(target_column, robust=self.robust)
        self.trend = stl_model.fit().trend
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result_df = df
        result_df[self.in_column] = self.trend
        return result_df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df
        result[self.in_column] = self.init_values
        return result


class STLTrendPerSegmentTransform(ReversiblePerSegmentWrapper):
    """Transform that changes target values to trend from STL decomposition"""

    def __init__(self, in_column: str, robust: bool):
        self.in_column = in_column
        self.robust = robust
        super().__init__(
            transform=_OneSegmentSTLTrend(in_column=self.in_column, robust=self.robust),
            required_features=[in_column]
        )

    # возможно тут надо дописать обратное преобразование

    def get_regressors_info(self) -> List[str]:
        return []


class _Driver:
    """
    Класс – интерфейс для классов PredictableDriver и OuterDriver
    
    Самый низкоуровневый класс, возвращающий в predict один временной ряд, 
    являющийся элементом в формуле декомпозиции целевой метрики
    """

    def __init__(
            self,
            df: pd.DataFrame,
            start_date=None,
            end_date=None,
            freq='D'
    ):
        self.driver_data = df
        self.name = df.columns[0]
        self.freq = pd.infer_freq(self.driver_data.index)

        assert self.freq is not None, "Wrong frequency"

        self.ts = TSDataset(self.driver_data, freq=self.freq)

    def __str__(self) -> str:
        return self.name

    def run_pipeline(self, plot_result=True, fit_best_model=False):
        pass

    def fit(self):
        pass

    def predict(self) -> TSDataset:
        pass


class PredictableDriver(_Driver):
    """
    Класс для драйвера, значения которого мы не знаем в будущем и будем предсказывать моделью
    """

    def __init__(self,
                 df: pd.DataFrame,
                 freq: str = 'D',
                 pipelines: List[Pipeline] = None,
                 preprocess: List[Transform] = None,
                 start_date=None,
                 end_date=None):

        super().__init__(df, start_date, end_date, freq)
        self.pipelines = deepcopy(pipelines)
        self.ts = TSDataset(self.driver_data, freq=self.freq)

        if preprocess:
            self.preprocess = preprocess
            self.ts.fit_transform(preprocess)

    def run_pipeline(self, plot_result=True, fit_best_model=False):
        self.mean_forecast_dfs = dict()
        self.mean_metrics = pd.DataFrame()

        for name, model in tqdm(self.pipelines.items()):
            # Temporarily suppress stderr with multiprocessing logs
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()

            mean_metrics_df, mean_forecast_df, _ = model.backtest(
                ts=self.ts, metrics=[SMAPE(), MAPE(), MSE()],
                aggregate_metrics=True, n_folds=3, joblib_params={'verbose': 0}
            )
            sys.stderr = old_stderr

            mean_metrics_df["model"] = name
            self.mean_metrics = pd.concat(
                [self.mean_metrics, mean_metrics_df.pivot(index='model', columns='segment')]
            )
            self.mean_forecast_dfs[name] = mean_forecast_df

        # self.best_model_name = self.mean_metrics.astype(int).iloc[:, :1].sum(axis=1).idxmin()
        # нормализуем СМАПЕ по каждому драйверу и выберем модель с минимальной суммой смапе по всем драйверам (для мультисегментной модели)
        mm = self.mean_metrics['SMAPE']
        mm = (mm - mm.min()) / (mm.max() - mm.min())
        self.best_model_name = mm.sum(axis=1).idxmin()

        # сам инстанс пайплайна должен быть не обучен, он только бэктестится! 
        # потом сохраняем копию лучшей по метрике модели и обучаем ее
        self.model = self.pipelines[self.best_model_name]
        print(f'Best score:\n{self.mean_metrics[self.mean_metrics.index == self.best_model_name]}\n')

        if plot_result:
            plot_backtest(self.mean_forecast_dfs[self.best_model_name], self.ts, history_len=60)

        if fit_best_model:
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            self.fit()
            sys.stderr = old_stderr

    def fit(self):
        if not self.model:
            self.run_pipeline(fit_best_model=True)
        self.model.fit(self.ts)

    def predict(self):
        if not self.model:
            raise Exception('Model is not fitted yet')
        # future_ts = self.ts.make_future(HORIZON)
        # трансформы применяются в модели, тк модель лежит в инстансе пайплайна
        # горизонт прогнозирования также лежит внутри трансформа
        # Temporarily suppress stderr with multiprocessing logs
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        forecast_ts = self.model.forecast()
        sys.stderr = old_stderr

        return forecast_ts


class OuterDriver(_Driver):
    """
    Класс для драйвера, значения которого мы знаем в будущем. Значения драйвера могут участвовать в финальной формуле подсчета GI.
    В библиотеке ETNA называется экзогенной фичой:
    - можем класть сюда пользовательские сценарии, например, значения предполагаемой маржи, индикатор возобновления онбординга новых клиентов
    - можем класть сюда другую известную о будущем инфу, например, запланированные акции по условиям тарифа
    """

    def __init__(
            self,
            df: pd.DataFrame,
            prediction_start_date: str,
            HORIZON: int,
            freq: str = 'D',
            start_date: str = None,
            end_date: str = None
    ):
        super().__init__(df, start_date, end_date, freq)
        self.prediction_start_date = prediction_start_date
        self.HORIZON = HORIZON

        if freq.startswith('D'):
            self.prediction_end_date = datetime.strftime(
                datetime.strptime(self.prediction_start_date, '%Y-%m-%d') + relativedelta(days=self.HORIZON),
                '%Y-%m-%d')
        if freq.startswith('W'):
            self.prediction_end_date = datetime.strftime(
                datetime.strptime(self.prediction_start_date, '%Y-%m-%d') + relativedelta(weeks=self.HORIZON),
                '%Y-%m-%d')
        if freq.startswith('M'):
            self.prediction_end_date = datetime.strftime(
                datetime.strptime(self.prediction_start_date, '%Y-%m-%d') + relativedelta(months=self.HORIZON),
                '%Y-%m-%d')

        self.best_model_name = self.name
        self.model = None

    def predict(self):
        df = self.ts.to_pandas()
        forecast = df[(df.index >= self.prediction_start_date) & (df.index < self.prediction_end_date)]
        return TSDataset(forecast, self.freq)


class GIComponent:
    """
    Компонента GI – второй по уровню класс для предсказания GI.
    Принимает:
    - список обученных драйверов
    - функцию для получения тотала из их прогнозов
    - булевый флаг, указывающий, надо ли занулять выходные дни
    """

    def __init__(self, drivers: list[_Driver], total_gi: Callable, set_zero_weekends=True):
        self.drivers = drivers
        self.name = self.drivers[0].name[0]
        self.total_gi = total_gi
        self.set_zero_weekends = set_zero_weekends

    def __str__(self):
        return self.name

    def fit_drivers(self):
        for _driver in self.drivers:
            _driver.fit()

    def switch_frequency(self, new_freq):
        for driver in self.drivers:
            # надо подумать где передать формулу для пересчета
            driver.change_frequency(self.drivers, new_freq)

    def calculate_total_gi(self):
        forecast = pd.DataFrame(self.total_gi(self.drivers))
        # зануляем выходные дни
        if self.set_zero_weekends:
            forecast.loc[pd.to_datetime(forecast.index).day_name().isin(['Saturday', 'Sunday']), 'target'] = 0
        return forecast['target']


class GI:
    def __init__(self, components: list[GIComponent]):
        self.components = components

    def fit_components(self):
        for component in self.components:
            component.fit_drivers()

    def calculate_total_gi(self):
        return np.sum(component.calculate_total_gi() for component in self.components)

    def switch_frequency(self, new_freq):
        # проходим по всем компонентам (вершинам дерева) и вызываем метод смены частоты, который задан у каждой из них
        pass

    def calculate_gi_per_component(self):
        result = pd.DataFrame()
        for component in self.components:
            component_forecast = pd.DataFrame(component.calculate_total_gi()).rename(columns={'target': component.name})
            result = pd.concat([result, component_forecast], axis=1)
        return result

    def calculate_gi_per_driver(self):
        result = pd.DataFrame()
        for component in self.components:
            for driver in component.drivers:
                forecast = driver.predict()[:, :, 'target']
                forecast.columns = [driver.name]
                # driver_forecast = pd.DataFrame(driver.predict()].iloc[:, 0]).rename(columns={'target': driver.name})
                result = pd.concat([result, forecast], axis=1)
        return result


def init_direct_pipelines(HORIZON: int, model_transforms: List[Transform]) -> List[Pipeline]:
    """
    Список пайплайнов для прогнозирования 1 временного ряда прямым подходом прогнозирования
    Все предсказания на HORIZON дней вперед совершаются моделью независимо, поэтому лаги не могут быть меньше чем HORIZON
    """
    params = {'transforms': model_transforms, 'horizon': HORIZON}
    pipelines = {
        "CatBoost": Pipeline(CatBoostPerSegmentModel(), **params),
        "Arima": Pipeline(AutoARIMAModel(), **params),
        'SimpleExpSmoothingModel': Pipeline(SimpleExpSmoothingModel(), **params),
        "Prophet": Pipeline(ProphetModel(), **params),
        "Linear": Pipeline(LinearPerSegmentModel(), **params),
        "LinearL1": Pipeline(ElasticPerSegmentModel(l1_ratio=1.0), **params),
        "LinearL2": Pipeline(ElasticPerSegmentModel(l1_ratio=0.0), **params),
        "RandomForest": Pipeline(SklearnPerSegmentModel(RandomForestRegressor(n_estimators=100)), **params),
        "RandomForestExtraTrees": Pipeline(SklearnPerSegmentModel(ExtraTreesRegressor(n_estimators=100)), **params)
    }
    ensebmle_models = [pipelines['CatBoost'], pipelines['RandomForestExtraTrees']]
    # усреднение прогнозов базовых моделей
    pipelines['VotingEnsemble'] = VotingEnsemble(ensebmle_models)
    # обученные веса на прогнозы базовых моделей
    pipelines['StackingEnsemble'] = StackingEnsemble(ensebmle_models)

    return pipelines


def init_autoreg_pipelines(HORIZON: int, model_transforms: List[Transform], step: int) -> List[Pipeline]:
    """
    Список пайплайнов для прогнозирования 1 временного ряда авторегрессионным подходом
    Может передавать лаги любого размера
    - HORIZON – горизонт прогнозирования
    – step – на сколько шагов вперед учится предсказывать модель
    То есть на 1 итерации предикта модель совершает horizon / step пересчетов признаков
    """
    params = {'transforms': model_transforms, 'horizon': HORIZON, 'step': step}
    pipelines = {
        "CatBoost": AutoRegressivePipeline(CatBoostPerSegmentModel(), **params),
        "Arima": AutoRegressivePipeline(AutoARIMAModel(), **params),
        'SimpleExpSmoothingModel': AutoRegressivePipeline(SimpleExpSmoothingModel(), **params),
        "Prophet": AutoRegressivePipeline(ProphetModel(), **params),
        "Linear": AutoRegressivePipeline(LinearPerSegmentModel(), **params),
        "LinearL1": AutoRegressivePipeline(ElasticPerSegmentModel(l1_ratio=1.0), **params),
        "LinearL2": AutoRegressivePipeline(ElasticPerSegmentModel(l1_ratio=0.0), **params),
        "RandomForest": AutoRegressivePipeline(SklearnPerSegmentModel(RandomForestRegressor(n_estimators=100)),
                                               **params),
        "RandomForestExtraTrees": AutoRegressivePipeline(SklearnPerSegmentModel(ExtraTreesRegressor(n_estimators=100)),
                                                         **params)
    }
    ensebmle_models = [pipelines['CatBoost'], pipelines['RandomForestExtraTrees']]
    # усреднение прогнозов базовых моделей
    pipelines['VotingEnsemble'] = VotingEnsemble(ensebmle_models)
    # обученные веса на прогнозы базовых моделей
    pipelines['StackingEnsemble'] = StackingEnsemble(ensebmle_models)

    return pipelines


def init_multiseg_direct_pipelines(HORIZON: int, model_transforms: List[Transform]) -> List[Pipeline]:
    """
    - HORIZON – горизонт прогнозирования
    """
    params = {'transforms': model_transforms, 'horizon': HORIZON}
    pipelines = {
        "CatBoost": Pipeline(CatBoostMultiSegmentModel(), **params),
        # "Arima": AutoRegressivePipeline(AutoARIMAModel(), **params),
        'SimpleExpSmoothingModel': Pipeline(SimpleExpSmoothingModel(), **params),
        # "Prophet": Pipeline(ProphetModel(), **params),
        "Linear": Pipeline(LinearMultiSegmentModel(), **params),
        "LinearL1": Pipeline(ElasticMultiSegmentModel(l1_ratio=1.0), **params),
        "LinearL2": Pipeline(ElasticMultiSegmentModel(l1_ratio=0.0), **params),
        "RandomForest": Pipeline(SklearnMultiSegmentModel(RandomForestRegressor(n_estimators=100)), **params),
        "RandomForestExtraTrees": Pipeline(SklearnMultiSegmentModel(ExtraTreesRegressor(n_estimators=100)), **params)
    }
    ensebmle_models = [pipelines['CatBoost'], pipelines['RandomForestExtraTrees']]
    # усреднение прогнозов базовых моделей
    pipelines['VotingEnsemble'] = VotingEnsemble(ensebmle_models)
    # обученные веса на прогнозы базовых моделей
    pipelines['StackingEnsemble'] = StackingEnsemble(ensebmle_models)

    return pipelines


def init_multiseg_autoreg_pipelines(HORIZON: int, model_transforms: List[Transform], step: int) -> List[Pipeline]:
    """
    - HORIZON – горизонт прогнозирования
    – step – на сколько шагов вперед учится предсказывать модель
    То есть на 1 итерации предикта модель совершает horizon / step пересчетов признаков
    """
    params = {'transforms': model_transforms, 'horizon': HORIZON, 'step': step}
    pipelines = {
        "CatBoost": AutoRegressivePipeline(CatBoostMultiSegmentModel(), **params),
        "Arima": AutoRegressivePipeline(AutoARIMAModel(), **params),
        'SimpleExpSmoothingModel': AutoRegressivePipeline(SimpleExpSmoothingModel(), **params),
        "Prophet": AutoRegressivePipeline(ProphetModel(), **params),
        "Linear": AutoRegressivePipeline(LinearMultiSegmentModel(), **params),
        "LinearL1": AutoRegressivePipeline(ElasticMultiSegmentModel(l1_ratio=1.0), **params),
        "LinearL2": AutoRegressivePipeline(ElasticMultiSegmentModel(l1_ratio=0.0), **params),
        "RandomForest": AutoRegressivePipeline(SklearnMultiSegmentModel(RandomForestRegressor(n_estimators=100)),
                                               **params),
        "RandomForestExtraTrees": AutoRegressivePipeline(
            SklearnMultiSegmentModel(ExtraTreesRegressor(n_estimators=100)), **params)
    }
    ensebmle_models = [pipelines['CatBoost'], pipelines['RandomForestExtraTrees']]
    # усреднение прогнозов базовых моделей
    pipelines['VotingEnsemble'] = VotingEnsemble(ensebmle_models)
    # обученные веса на прогнозы базовых моделей
    pipelines['StackingEnsemble'] = StackingEnsemble(ensebmle_models)

    return pipelines
