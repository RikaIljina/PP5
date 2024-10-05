from src.data_management import load_pkl_file


def load_test_evaluation():
    return (
        load_pkl_file(f'outputs/x_b_1/model_final_eval_x_b_1.pkl'),
        load_pkl_file(f'outputs/x_b_1/hyperparam_values_x_b_1.pkl'),
    )


def load_reports():
    return {'test': load_pkl_file(
                    f'outputs/x_b_1/test_class_report_x_b_1.pkl'),
            'live': load_pkl_file(
                    f'outputs/x_b_1/live_class_report_x_b_1.pkl'),
            'live_batches': load_pkl_file(
                    f'outputs/x_b_1/live_class_report_batches_x_b_1.pkl'),
            }
