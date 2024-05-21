import os

import matplotlib.pyplot as plt
import seaborn as sns
from etna.analysis import plot_forecast

from architecture import *
from architecture import _Driver

sns.set_theme()


def train_test_split(df, train_start, test_start, test_end):
    train_df = df[(df.val_date >= train_start) & (df.val_date < test_start)]
    test_df = df[(df.val_date >= test_start) & (df.val_date <= test_end)]
    return train_df, test_df


def write_train_test_csv(drivers_path, train_path, test_path, train_start, test_start, test_end):
    for filename in os.listdir(drivers_path):
        if not filename.endswith('csv'):
            continue
        print(f'PROCESS {filename}')
        df = pd.read_csv(f'{drivers_path}/{filename}')
        train_df, test_df = train_test_split(df, train_start, test_start, test_end)
        train_df.to_csv(f'{train_path}/{filename}', index=False)
        test_df.to_csv(f'{test_path}/{filename}', index=False)


def mul_drivers(drivers_list: List[_Driver]):
    predictions = np.ones(drivers_list[0].predict().to_pandas().shape[0])

    for driver in drivers_list:
        forecast = driver.predict().to_pandas()
        forecast.columns = forecast.columns.get_level_values('feature')
        predictions *= forecast['target']

    predictions.index = pd.to_datetime(predictions.index)
    return predictions


def set_freq(df, freq):
    """
    При наличии отсутствия записей на какие-то даты (например, комиссии по выходным)
    дополняем датафрейм до ежедневной частоты, заполняя пропуски лагами
    """
    min_date = df.index.min()
    max_date = df.index.max()

    dates = (
        pd.DataFrame(pd.date_range(start=min_date, end=max_date, freq=freq))
        .rename(columns={0: 'timestamp'})
        .set_index('timestamp')
    )
    df = dates.join(df)
    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')
    return df


def collect_data(path, start_date=None, end_date=None, freq='D'):
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    df.index.name = 'timestamp'
    df = set_freq(df, freq)

    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]

    return df


def collect_driver(path, start_date=None, end_date=None, freq='D'):
    """
    Базовая функция, считывающая файл с драйверами и приводящая названия колонок в формат etna.TSDataset
    """
    df = collect_data(path, start_date=start_date, end_date=end_date, freq=freq)
    df.columns = pd.MultiIndex.from_tuples(tuple(map(lambda x: (x, 'target'), df.columns)),
                                           name=('segment', 'feature'))
    return df


def collect_all_drivers(path, start_date=None, end_date=None, freq='D'):
    df = collect_data(path, start_date=start_date, end_date=end_date, freq=freq)
    df.columns = pd.MultiIndex.from_tuples(tuple(map(lambda x: (x, 'target'), df.columns)),
                                           name=('segment', 'feature'))
    df = df.drop(columns=['702226_rate'])
    return df


def get_report(test_path, driver, collect_driver: Callable, apply_preprocess_to_test=True, freq='D'):
    test_df = collect_driver(test_path, freq=freq)
    test_ts = TSDataset(test_df, freq=pd.infer_freq(test_df.index))

    print(f'Best model: {driver.best_model_name}')
    forecast_ts = driver.predict()
    smape, mape, mse = SMAPE(), MAPE(), MSE()
    print("SMAPE:\t", smape(test_ts, forecast_ts))
    print('MAPE:\t', mape(test_ts, forecast_ts))
    print('MSE:\t', mse(test_ts, forecast_ts))
    plot_forecast(forecast_ts, test_ts, driver.ts, n_train_samples=45)


def collect_target_per_component(target_dir, groupby_field='val_date'):
    target = pd.DataFrame()
    for file in os.listdir(target_dir):
        if file.endswith('.csv'):
            df = pd.read_csv(f'{target_dir}/{file}')
            acod = df.acod.unique()[0]
            df = df.rename(columns={'gi': acod}).drop(columns=['acod'])
            if target.shape[0] > 0:
                target = pd.merge(df, target, on=groupby_field, how='outer')
            else:
                target = df

    target = target.groupby(groupby_field).sum().reset_index()
    target = target.set_index(groupby_field).sort_index()
    target.columns = target.columns.astype(str)
    target.index = pd.to_datetime(target.index)
    target.index.name = groupby_field

    return target


def collect_target_per_driver(target_dir, filename, drop_cols=[]):
    target_per_driver = pd.read_csv(f'{target_dir}/{filename}').drop(columns=drop_cols)
    target_per_driver.val_date = pd.to_datetime(target_per_driver.val_date)
    target_per_driver = target_per_driver.set_index('val_date')
    return target_per_driver


def collect_target_total(target_dir, groupby_field='val_date'):
    return collect_target_per_component(target_dir, groupby_field=groupby_field).sum(axis=1)


def build_components(component_names: List[str], drivers_list: List[_Driver], total_gi: Callable,
                     set_zero_weekends: bool = False) -> GI:
    """
    Принимает
    - component_names: список названий компонентов
    - drivers_list: список обученных драйверов (умеют делать predict)
    – total_gi: функция для подсчета GI в одной компоненте – ЕДИНАЯ ДЛЯ ВСЕХ ПАР ДРАЙВЕРОВ
    - set_zero_weekends: булевый индиктор – зануляем ли выходные дни
    """
    components = []

    for component in component_names:
        component_drivers = list(filter(lambda x: x.name.startswith(component), drivers_list))
        gi_component = GIComponent(drivers=list(component_drivers), total_gi=total_gi,
                                   set_zero_weekends=set_zero_weekends)
        components.append(gi_component)

    GI_predictor = GI(components)
    return GI_predictor


def collect_driver_from_multiseg(multiseg_ts, start_date=None, end_date=None, freq='D'):
    """
    Подготовка датафрейма из мультсегментной модели к иниту объекту драйвера на каждом сегменте
    """
    multiseg_ts = set_freq(multiseg_ts, freq)
    if start_date:
        multiseg_ts = multiseg_ts[multiseg_ts.index >= start_date]
    if end_date:
        multiseg_ts = multiseg_ts[multiseg_ts.index <= end_date]
    multiseg_ts.columns = pd.MultiIndex.from_tuples(tuple(map(lambda x: (x, 'target'), multiseg_ts.columns)),
                                                    name=('segment', 'feature'))
    return multiseg_ts


def plot_total_prediction(target_total: pd.DataFrame, preds_total: pd.DataFrame, test_start: str, test_end: str,
                          n_train_samples=30):
    plt.figure(figsize=(12, 6))
    plt.plot(target_total[(target_total.index <= test_start)].tail(n_train_samples), label='target train')
    plt.plot(target_total[(target_total.index >= test_start) & (target_total.index <= test_end)], label='target test')
    plt.plot(preds_total, label='pred')
    plt.ylim(0)
    plt.title('Total GI target vs predictions')
    plt.xticks(rotation=30)
    plt.legend()
    plt.show()


def plot_prediction_detailed(target_per_driver, preds_per_driver, test_start: str, test_end: str, n_train_samples=30):
    n_drivers = len(target_per_driver.columns)
    n_cols = 3
    n_rows = n_drivers // n_cols + 1

    f, axes = plt.subplots(n_rows, n_cols, figsize=(25, 12))

    for i in range(n_rows):
        for j in range(n_cols):
            if len(target_per_driver.columns) <= (i * n_cols + j):
                break
            driver = target_per_driver.columns[i * n_cols + j]
            ax = axes[i, j]

            ax.plot(target_per_driver[(target_per_driver.index <= test_start)][driver].tail(n_train_samples),
                    label='target train')
            ax.plot(target_per_driver[(target_per_driver.index >= test_start) & (target_per_driver.index <= test_end)][
                        driver], label='target test')
            ax.plot(preds_per_driver[driver], label='pred', color='g')
            ax.set_title(driver)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')
            ax.legend()
    plt.tight_layout()


def my_smape(forecast, target):
    return np.mean(np.abs(forecast - target) / (0.5 * (np.abs(forecast) + np.abs(target)))) * 100
