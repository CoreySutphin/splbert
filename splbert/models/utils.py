from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn_crfsuite import metrics
from tabulate import tabulate
from statistics import mean


def cross_validate(model, X, y, k=5):
    """Perform K-Fold Stratified Cross Validation using a given model."""
    skf = StratifiedKFold(n_splits=k)
    mlb = MultiLabelBinarizer()
    print(X)
    print(y)

    for train_index, test_index in skf.split(
        mlb.fit_transform(X), mlb.fit_transform(y)
    ):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train using the provided data
        model.fit(x_train, y_train)

        # Generate predictions using the trained model
        y_pred = model.predict(x_test)

        # Calculate performance metrics for this fold(Precision, Recall, F1)
        # Write the metrics for this fold.
        for label in tags:
            fold_statistics[label] = {
                "recall": metrics.flat_recall_score(
                    y_test, y_pred, average="weighted", labels=[label]
                ),
                "precision": metrics.flat_precision_score(
                    y_test, y_pred, average="weighted", labels=[label]
                ),
                "f1": metrics.flat_f1_score(
                    y_test, y_pred, average="weighted", labels=[label]
                ),
            }

        # add averages
        fold_statistics["system"] = {
            "recall": metrics.flat_recall_score(
                y_test, y_pred, average="weighted", labels=tags
            ),
            "precision": metrics.flat_precision_score(
                y_test, y_pred, average="weighted", labels=tags
            ),
            "f1": metrics.flat_f1_score(
                y_test, y_pred, average="weighted", labels=tags
            ),
        }

        table_data = [
            [
                label,
                format(fold_statistics[label]["precision"], ".3f"),
                format(fold_statistics[label]["recall"], ".3f"),
                format(fold_statistics[label]["f1"], ".3f"),
            ]
            for label in tags + ["system"]
        ]

        logging.info(
            "\n"
            + tabulate(
                table_data,
                headers=["Entity", "Precision", "Recall", "F1"],
                tablefmt="orgtbl",
            )
        )

        eval_stats[fold_num] = fold_statistics

    statistics_all_folds = {}

    for label in tags + ["system"]:
        statistics_all_folds[label] = {
            "precision_average": mean(
                eval_stats[fold][label]["precision"] for fold in eval_stats
            ),
            "precision_max": max(
                eval_stats[fold][label]["precision"] for fold in eval_stats
            ),
            "precision_min": min(
                eval_stats[fold][label]["precision"] for fold in eval_stats
            ),
            "recall_average": mean(
                eval_stats[fold][label]["recall"] for fold in eval_stats
            ),
            "recall_max": max(eval_stats[fold][label]["recall"] for fold in eval_stats),
            "f1_average": mean(eval_stats[fold][label]["f1"] for fold in eval_stats),
            "f1_max": max(eval_stats[fold][label]["f1"] for fold in eval_stats),
            "f1_min": min(eval_stats[fold][label]["f1"] for fold in eval_stats),
        }

    entity_counts = training_dataset.compute_counts()
    entity_counts["system"] = sum(
        v for k, v in entity_counts.items() if k in self.pipeline.entities
    )

    table_data = [
        [
            f"{label} ({entity_counts[label]})",  # Entity (Count)
            format(statistics_all_folds[label]["precision_average"], ".3f"),
            format(statistics_all_folds[label]["recall_average"], ".3f"),
            format(statistics_all_folds[label]["f1_average"], ".3f"),
            format(statistics_all_folds[label]["f1_min"], ".3f"),
            format(statistics_all_folds[label]["f1_max"], ".3f"),
        ]
        for label in tags + ["system"]
    ]

    # Combine the pipeline report and the resulting data, then log it or print it (whichever ensures that it prints)

    output_str = (
        "\n"
        + pipeline_report
        + "\n\n"
        + tabulate(
            table_data,
            headers=["Entity (Count)", "Precision", "Recall", "F1", "F1_Min", "F1_Max"],
            tablefmt="orgtbl",
        )
    )

    if logging.root.level > logging.INFO:
        print(output_str)
    else:
        logging.info(output_str)

    return
