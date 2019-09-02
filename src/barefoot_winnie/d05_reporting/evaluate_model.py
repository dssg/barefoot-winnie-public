import numpy as np
from barefoot_winnie.d00_utils.metrics_utils import produce_similarity_score_text
import matplotlib.pyplot as plt
import pandas as pd


def evaluate_model(model_results,
                   metrics=['overlap', 'jaro', 'jaccard']):
    """ Calculates algorithmic metrics for Winnie recommendations
    :param model_results: Pandas dataframe with predicted responses and the true responses
    :param metrics: list of metrics to be calculated
    :return: Performance measures for each question, a summary and plots
    """

    for metric_ in metrics:
        model_results[metric_] = model_results.apply(lambda x: produce_similarity_score_text(
                                                     x['response'],
                                                     x['true_response'], metric=metric_), axis=1)

    model_results['length'] = model_results.question.str.len()
    model_results['first_message_flag'] = np.where(model_results['num_of_messages'] == 1, True, False)
    recommendation_performance = model_results[['interaction_id',
                                                'channel',
                                                'length',
                                                'first_message_flag',
                                                'rank']
                                               + metrics]

    performance_per_interaction_avg = (recommendation_performance
                                       .groupby(['interaction_id', 'channel', 'length', 'first_message_flag'])
                                       [metrics]
                                       .mean()
                                       .reset_index()
                                       .rename(columns={metric: 'avg_' + metric for metric in metrics}))

    performance_per_interaction_top = (recommendation_performance.loc[model_results['rank'] == 1]
                                       .groupby('interaction_id')
                                       [metrics]
                                       .mean()
                                       .reset_index()
                                       .rename(columns={metric: 'rank_1_' + metric for metric in metrics})
                                       )
    performance_per_interaction = pd.merge(left=performance_per_interaction_avg,
                                           right=performance_per_interaction_top,
                                           on=['interaction_id'],
                                           how='inner')

    performance_plots = {}

    bins = np.linspace(0, 1, 50)

    report_name = 'performance_per_interaction'
    avg_metrics = ['avg_' + metric for metric in metrics]
    top_metrics = ['rank_1_' + metric for metric in metrics]

    for metric in avg_metrics:
        plt.hist(performance_per_interaction[performance_per_interaction['first_message_flag']][metric],
                 bins,
                 alpha=0.5,
                 label='First interaction')
        plt.hist(performance_per_interaction[~performance_per_interaction['first_message_flag']][metric],
                 bins,
                 alpha=0.5,
                 label='Follow up')
        plt.legend(loc='upper right')
        plt.title(metric + '_' + report_name + '_first_vs_followup')
        plt.xlabel(metric)
        plt.ylabel('number of messages')
        performance_plots[metric + '_' + report_name + '_first_vs_followup'] = plt.figure()

    for metric in avg_metrics:
        plt.hist(performance_per_interaction[performance_per_interaction['channel'] == 'Facebook'][metric],
                 bins,
                 alpha=0.5,
                 label='Facebook')
        plt.hist(performance_per_interaction[performance_per_interaction['channel'] == 'SMS'][metric],
                 bins,
                 alpha=0.5,
                 label='SMS')
        plt.legend(loc='upper right')
        plt.title(metric + '_' + report_name + '_channel_first_vs_followup')
        plt.xlabel(metric)
        plt.ylabel('number of messages')
        performance_plots[metric + '_' + report_name + '_channel_first_vs_followup'] = plt.figure()

    # length vs performance per channel
    channel_dict = {'Facebook': 'fb',
                    'SMS': 'sms'}

    for channel, channel_nickname in channel_dict.items():
        channel_performance = performance_per_interaction.loc[performance_per_interaction['channel'] == channel]
        if not channel_performance.empty:
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 4))
            for xcol, ax in zip(avg_metrics, axes):
                plot = channel_performance.plot(kind='scatter',
                                                y=xcol,
                                                x='length',
                                                ax=ax,
                                                alpha=0.5, color='r', xlim=[0, 1000],
                                                title=report_name + ' ' + channel + '\n Score vs. Length')
            performance_plots[report_name + '_' + channel_nickname + '_score_vs_length'] = plot.figure

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14,4))
    for xcol, ax in zip(avg_metrics, axes):
        plot = performance_per_interaction.plot(kind='scatter',
                                                y=xcol,
                                                x='length',
                                                ax=ax,
                                                alpha=0.5,
                                                color='r',
                                                xlim=[0, 1000],
                                                title=report_name + '\n all score vs length')
    performance_plots[report_name + '_all_score_vs_length'] = plot.figure

    colours = ['r', 'g', 'b']
    for metric, colour in zip(avg_metrics, colours):
        plt.scatter(performance_per_interaction['length'],
                    performance_per_interaction[metric],
                    color=colour,
                    label=metric)
    plt.xlabel('Question length')
    plt.ylabel('Scores')
    plt.legend()
    plt.title(report_name + '_all_score_vs_length_separate')
    performance_plots[report_name + '_all_score_vs_length_separate'] = plt.figure()

    performance_cols = avg_metrics + top_metrics

    # overall summary
    all_summary = pd.DataFrame(performance_per_interaction[performance_cols].mean(axis=0, skipna=True)).T
    all_summary['question_subset'] = 'all'

    # channel specific
    channel_summary = (performance_per_interaction
                       .groupby('channel')
                       [performance_cols]
                       .mean()
                       .reset_index()
                       .rename(columns={'channel': 'question_subset'}))

    # only first
    only_fst_question = pd.DataFrame(performance_per_interaction
                                     .loc[performance_per_interaction['first_message_flag'],
                                          performance_cols].mean(axis=0, skipna=True)).T
    only_fst_question['question_subset'] = 'only_first'

    # combine
    performance_summary = pd.concat([all_summary,
                                     channel_summary,
                                     only_fst_question],
                                    sort=True)
    performance_summary = performance_summary[['question_subset'] + performance_cols]

    return [recommendation_performance, performance_summary, performance_plots]
