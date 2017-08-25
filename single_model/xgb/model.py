import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import xgboost as xgb

def xgb0(df_cell_train_feats, y_train, df_cell_test_feats):
    def prepare_feats(df):
        return df.drop(['time'], axis=1)
    logging.info("train xgb0 model")
    clf = xgb.XGBClassifier()
    clf.fit(prepare_feats(df_cell_train_feats), y_train)
    y_test_pred = clf.predict_proba(prepare_feats(df_cell_test_feats))
    return y_test_pred


model_dict = {"xgb0": xgb0}

if __name__ == "__main__":
    config_name = sys.argv[1]
    data_path = "../../data/"
run_model(config_name, model_dict, data_path)