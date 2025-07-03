from sqlalchemy import create_engine, MetaData, Table, select, desc, asc, and_, or_, between, cast, Float, Integer, String
import yaml
import pandas as pd
from tabulate import tabulate
from typing import Dict, Any
from dotenv import load_dotenv
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

load_dotenv("db.env")

postgres_user = os.environ["POSTGRES_USER"]
postgres_password = os.environ["POSTGRES_PASSWORD"]
postgres_host = os.environ["POSTGRES_HOST"]
postgres_port = os.environ["POSTGRES_PORT"]
postgres_db = os.environ["POSTGRES_DB"]

def validate_config(config: Dict[str, Any]) -> None:
    """Validate YAML configuration structure and types"""
    type_map = {'float': float, 'int': int, 'str': str}

    for param in config.get('parameters_to_include', []):
        if 'type' not in param:
            raise ValueError(f"Missing type for parameter: {param.get('name')}")
        if param['type'] not in type_map:
            raise ValueError(f"Invalid type for {param['name']}: {param['type']}")

def load_config():
    with open('./src/config/metrics_config.yaml') as f:
        return yaml.safe_load(f)

def get_tracking_uri():
    return f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"

def build_experiment_filter(experiments_table, config):
    filter_conf = config.get('experiment_filter', {})
    conditions = []
    if 'include' in filter_conf and 'experiment_names' in filter_conf['include']:
        conditions.append(
            experiments_table.c.name.in_(filter_conf['include']['experiment_names'])
        )
    return or_(*conditions) if conditions else True

def build_parameter_filters(params_table, config):
    """Build parameter filters from config with comprehensive null checks"""
    conditions = []
    type_map = {'float': Float, 'int': Integer, 'str': String}

    # Safely get parameter_filters with empty dict as default
    param_filters = config.get('parameter_filters', {}) or {}

    # Process numeric ranges if section exists and has content
    numeric_ranges = param_filters.get('numeric_ranges')
    if isinstance(numeric_ranges, dict):  # Explicit type check
        for param, ranges in numeric_ranges.items():
            # Skip if ranges is None or not a dict
            if not isinstance(ranges, dict):
                continue

            # Get parameter type (default to float)
            param_type = next(
                (p['type'] for p in config.get('parameters_to_include', [])
                 if p['name'] == param),
                'float'
            )
            col = cast(params_table.c.value, type_map[param_type])

            # Handle range filters - min/max are optional
            range_conditions = []
            if 'min' in ranges:
                range_conditions.append(col >= ranges['min'])
            if 'max' in ranges:
                range_conditions.append(col <= ranges['max'])

            if range_conditions:
                conditions.append(
                    and_(params_table.c.key == param, *range_conditions)
                )

    # Process categorical filters if section exists and has content
    categorical_filters = param_filters.get('categorical')
    if isinstance(categorical_filters, dict):
        for param, settings in categorical_filters.items():
            # Skip if settings is None or not a dict
            if not isinstance(settings, dict):
                continue

            values = settings.get('values')
            if isinstance(values, (list, tuple)):
                mode = settings.get('mode', 'include')
                conditions.append(
                    and_(
                        params_table.c.key == param,
                        params_table.c.value.in_(values) if mode == 'include'
                        else ~params_table.c.value.in_(values)
                    )
                )

    return conditions

def generate_statistics(df: pd.DataFrame, config: Dict) -> str:
    """Generate formatted statistics for numeric and categorical parameters"""
    stats = ["\n=== PARAMETER STATISTICS ==="]

    # Get parameter definitions from config
    param_defs = {p['name']: p['type'] for p in config['parameters_to_include']}

    # Numeric statistics
    numeric_params = [name for name, dtype in param_defs.items() if dtype in ['float', 'int']]
    if numeric_params and any(p in df.columns for p in numeric_params):
        stats.append("\nNumeric Parameters:")
        num_stats = df[numeric_params].apply(pd.to_numeric, errors='coerce').describe(percentiles=[.25, .5, .75])
        stats.append(tabulate(num_stats, headers='keys', floatfmt=".3f"))

    # Categorical statistics
    categorical_params = [name for name, dtype in param_defs.items() if dtype == 'str']
    if categorical_params and any(p in df.columns for p in categorical_params):
        stats.append("\nCategorical Parameters:")
        cat_stats = []
        for param in categorical_params:
            if param in df.columns:
                counts = df[param].value_counts(dropna=False)
                cat_stats.append({
                    'Parameter': param,
                    'Unique Values': len(counts),
                    'Top Value': f"{counts.index[0]} (n={counts.iloc[0]})",
                    'Null Values': df[param].isna().sum()
                })
        if cat_stats:
            stats.append(tabulate(pd.DataFrame(cat_stats), headers='keys', showindex=False))

    return "\n".join(stats)

def generate_correlation_analysis(df: pd.DataFrame, config: Dict) -> str:
    """Generate correlation analysis between numeric parameters and metrics"""
    analysis = ["\n=== METRIC-PARAMETER CORRELATION ==="]

    numeric_params = [
        p['name'] for p in config['parameters_to_include']
        if p['type'] in ['float', 'int'] and p['name'] in df.columns
    ]
    metrics = [m['name'] for m in config['metrics_to_sort'] if m['name'] in df.columns]

    if not numeric_params or not metrics:
        return "\nNo numeric parameters/metrics for correlation analysis"

    # Convert to numeric and calculate correlations
    corr_df = df[numeric_params + metrics].apply(pd.to_numeric, errors='coerce')
    corr_matrix = corr_df.corr(method='pearson')

    # Get strong correlations
    threshold = config.get('output', {}).get('correlation_threshold', 0.3)
    strong_corrs = []

    for param in numeric_params:
        for metric in metrics:
            corr = corr_matrix.loc[param, metric]
            if abs(corr) >= threshold:
                direction = "POSITIVE" if corr > 0 else "NEGATIVE"
                strong_corrs.append({
                    'Parameter': param,
                    'Metric': metric,
                    'Correlation': f"{corr:.2f} ({direction})"
                })

    if strong_corrs:
        analysis.append(tabulate(pd.DataFrame(strong_corrs), headers='keys', showindex=False))
    else:
        analysis.append(f"No strong correlations (|r| > {threshold}) found")

    # Generate correlation plot
    if config.get('output', {}).get('plots_dir'):
        plt.ioff()  # Turn off interactive mode
        try:
            os.makedirs(config['output']['plots_dir'], exist_ok=True)
            plt.figure(figsize=(10, 6))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Parameter-Metric Correlations')
            plot_path = os.path.join(config['output']['plots_dir'], 'correlation_heatmap.png')
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            analysis.append(f"Correlation plot saved to: {plot_path}")
        except Exception as e:
            analysis.append(f"Failed to save plot: {str(e)}")

    return "\n".join(analysis)

def save_report(results: pd.DataFrame, analysis: str, config: Dict):
    """Save full report to file"""
    if not config.get('output', {}).get('report_file'):
        return

    filename = config['output']['report_file'].format(
        date=datetime.now().strftime('%Y%m%d')
    )

    print(f"Saving full report to: {filename}")

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as f:
        f.write("=== MODEL TRAINING REPORT ===\n")
        f.write(f"Generated: {datetime.now()}\n\n")
        f.write("=== MODEL RESULTS ===\n")
        f.write(results.to_markdown(index=False))
        f.write("\n\n")
        f.write(analysis)

    print(f"\nFull report saved to: {filename}")

def get_sorted_models(config):
    try:
        validate_config(config)
    except ValueError as e:
        print(f"Configuration error: {e}")
        return pd.DataFrame()

    engine = create_engine(get_tracking_uri())
    metadata = MetaData()

    experiments = Table('experiments', metadata, autoload_with=engine)
    runs = Table('runs', metadata, autoload_with=engine)
    metrics = Table('metrics', metadata, autoload_with=engine)
    params = Table('params', metadata, autoload_with=engine)

    # First get all runs that match our parameter filters
    param_conditions = build_parameter_filters(params, config)
    filtered_runs = set()

    if param_conditions:
        with engine.connect() as conn:
            param_query = select(params.c.run_uuid.distinct()).where(or_(*param_conditions))
            filtered_runs = {row[0] for row in conn.execute(param_query)}
            if not filtered_runs:
                print("No runs matched the parameter filters")
                return pd.DataFrame()

    # Main query for metrics
    order_by = [
        desc(metrics.c.value) if not m['ascending'] else asc(metrics.c.value)
        for m in config['metrics_to_sort']
    ]

    base_query = select(
        experiments.c.name.label('experiment_name'),
        runs.c.run_uuid,
        runs.c.name.label('run_name'),
        *[metrics.c.value.label(m['name']) for m in config['metrics_to_sort']]
    ).select_from(
        runs.join(experiments, runs.c.experiment_id == experiments.c.experiment_id)
            .join(metrics, metrics.c.run_uuid == runs.c.run_uuid)
    ).where(
        and_(
            metrics.c.key.in_([m['name'] for m in config['metrics_to_sort']]),
            build_experiment_filter(experiments, config),
            runs.c.run_uuid.in_(filtered_runs) if filtered_runs else True
        )
    ).order_by(*order_by)

    # Get parameters for display
    param_names = [p['name'] for p in config.get('parameters_to_include', [])]
    param_query = select(
        params.c.run_uuid,
        params.c.key,
        params.c.value
    ).where(
        and_(
            params.c.key.in_(param_names),
            params.c.run_uuid.in_(filtered_runs) if filtered_runs else True
        )
    )

    metrics_df = pd.read_sql(base_query, engine)

    if not metrics_df.empty:
        params_df = pd.read_sql(param_query, engine)
        if not params_df.empty:
            params_pivot = params_df.pivot(
                index='run_uuid',
                columns='key',
                values='value'
            ).reset_index()
            metrics_df = pd.merge(metrics_df, params_pivot, on='run_uuid', how='left')

        # Generate and print statistics
        analysis = []
        analysis.append(generate_statistics(metrics_df, config))
        analysis.append(generate_correlation_analysis(metrics_df, config))
        full_analysis = "\n".join(analysis)

        print(full_analysis)

        save_report(metrics_df, full_analysis, config)

    return metrics_df

if __name__ == "__main__":
    config = load_config()
    results = get_sorted_models(config)
    print(results.to_markdown(index=False))
