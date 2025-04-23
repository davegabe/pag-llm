import wandb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re


def create_wandb_custom_visualization(run_path: str):
    """
    Create custom visualizations in a wandb run.

    Args:
        run_path: Path to wandb run in format "entity/project/run_id"
    """
    # Initialize wandb API
    api = wandb.Api()

    # Get the run
    run = api.run(run_path)
    print(run)

    # Get metrics
    metrics = []

    for key, value in run.summary.items():
        if key.startswith('test/m_'):
            pattern = r'test/m_(\d+)_t_(\d+)/(.+)'
            match = re.match(pattern, key)

            if match:
                mask_value = int(match.group(1))
                token_count = int(match.group(2))
                metric_type = match.group(3)

                metrics.append({
                    'mask_value': mask_value,
                    'token_count': token_count,
                    'metric_type': metric_type,
                    'value': value
                })

    df = pd.DataFrame(metrics)

    if df.empty:
        print("No test metrics found")
        return

    # Create interactive visualizations with plotly
    mask_values = sorted(df['mask_value'].unique())

    for mask_value in mask_values:
        mask_df = df[df['mask_value'] == mask_value]

        # Create subplot with 2 rows
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f"Accuracies for Mask {mask_value}",
            f"Loss for Mask {mask_value}")
        )

        # Add traces for top-k accuracies
        for metric in ['top_1_acc', 'top_2_acc', 'top_3_acc']:
            data = mask_df[mask_df['metric_type'] == metric].sort_values('token_count')
            if not data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=data['token_count'],
                        y=data['value'],
                        mode='lines+markers',
                        name=metric
                    ),
                    row=1, col=1
                )

        # Add trace for loss
        loss_data = mask_df[mask_df['metric_type'] == 'loss_inv'].sort_values('token_count')
        if not loss_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=loss_data['token_count'],
                    y=loss_data['value'],
                    mode='lines+markers',
                    name='loss_inv',
                    line=dict(color='red')
                ),
                row=2, col=1
            )

        # Update layout
        fig.update_layout(
            title=f"Metrics for Mask Value {mask_value}",
            height=800,
            width=1000
        )

        # Create a new version of the run to add these plots
        with wandb.init(id=run.id) as resumed_run:
            resumed_run.log({f"interactive_plots/mask_{mask_value}": fig})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--run_path', type=str, required=True,
        help='Path to wandb run in format "entity/project/run_id"'
    )
    args = parser.parse_args()

    create_wandb_custom_visualization(args.run_path)
