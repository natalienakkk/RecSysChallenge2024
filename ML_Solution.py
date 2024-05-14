"""
 Natalie Nakkara , 16/05/2024
"""
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, average_precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import ADASYN

def load_data(path):
    """
    Load behavior and history data from specified path.

    Parameters:
    path (str): Path to the dataset

    """
    behaviors = pd.read_parquet(f'{path}/behaviors.parquet')
    history = pd.read_parquet(f'{path}/history.parquet')
    return behaviors, history

def preprocess_data(behaviors, articles):
    """
    Preprocess the behavior and article data.

    """
    # Convert 'impression_time' to datetime and extract hour and day of the week
    behaviors['hour_of_day'] = pd.to_datetime(behaviors['impression_time']).dt.hour
    behaviors['day_of_week'] = pd.to_datetime(behaviors['impression_time']).dt.dayofweek

    # Calculate behavioral features
    user_avg_read_time = behaviors.groupby('user_id')['read_time'].mean().rename('user_avg_read_time')
    user_avg_scroll_percentage = behaviors.groupby('user_id')['scroll_percentage'].mean().rename('user_avg_scroll_percentage')
    user_total_clicks = behaviors.groupby('user_id')['article_id'].count().rename('user_total_clicks')
    behaviors = behaviors.join(user_avg_read_time, on='user_id')
    behaviors = behaviors.join(user_avg_scroll_percentage, on='user_id')
    behaviors = behaviors.join(user_total_clicks, on='user_id')

    # Process article data
    articles['year_published'] = pd.to_datetime(articles['published_time']).dt.year
    articles['year_modified'] = pd.to_datetime(articles['published_time']).dt.year
    articles['article_age'] = (pd.to_datetime('now') - pd.to_datetime(articles['published_time'])).dt.days

    label_encoder = LabelEncoder()
    articles['category_str'] = label_encoder.fit_transform(articles['category_str'])
    articles['article_type'] = label_encoder.fit_transform(articles['article_type'])
    articles['sentiment_label'] = label_encoder.fit_transform(articles['sentiment_label'])

    articles = articles[['article_id', 'category_str', 'category', 'year_published', 'year_modified', 'article_type',
                         'total_pageviews', 'sentiment_label', 'article_age']]

    behaviors = behaviors.merge(articles, on='article_id', how='left')

    # Impute missing values
    num_features = behaviors.select_dtypes(include=[np.number]).columns
    num_imputer = SimpleImputer(strategy='median')
    behaviors[num_features] = num_imputer.fit_transform(behaviors[num_features])

    return behaviors

def train_model(features, target):
    """
    Train an XGBoost model.

    Parameters:
    features DataFrame: Training features
    target Series: Training target

    Returns:
    model XGBClassifier: Trained XGBoost model
    """
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.2,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42,
    )
    scores = cross_val_score(model, features, target, cv=5)
    print("Cross-validated accuracy scores:", scores)
    print("Average cross-validation score:", np.mean(scores))
    model.fit(features, target)
    return model

def evaluate_model(model, features, target):
    """
    Evaluate the model's performance.

    Parameters:
    model (XGBClassifier): Trained XGBoost model
    features (DataFrame): Features to evaluate
    target (Series): labeled data

    """
    predictions = model.predict(features)
    prediction_probs = model.predict_proba(features)[:, 1]

    acc = accuracy_score(target, predictions)
    print("Accuracy:", acc)

    class_report = classification_report(target, predictions)
    print("Classification Report:")
    print(class_report)

    roc_auc = roc_auc_score(target, prediction_probs)
    print("ROC AUC Score:", roc_auc)

    pr_auc = average_precision_score(target, prediction_probs)
    print("PR AUC Score:", pr_auc)

def expand_and_label(behaviors):
    """
    Expand and label behavior data.

    """
    expanded_rows = []
    for _, row in behaviors.iterrows():
        articles_in_view = row['article_ids_inview']
        for article_id in articles_in_view:
            expanded_row = {
                'user_id': row['user_id'],
                'article_id': article_id,
                'hour_of_day': row['hour_of_day'],
                'day_of_week': row['day_of_week'],
                'device_type': row['device_type'],
                'is_subscriber': row['is_subscriber'],
                'read_time': row['read_time'],
                'scroll_percentage': row['scroll_percentage'],
                'was_clicked': 1 if article_id in row['article_ids_clicked'] else 0,
                'user_avg_read_time': row['user_avg_read_time'],
                'user_avg_scroll_percentage': row['user_avg_scroll_percentage'],
                'user_total_clicks': row['user_total_clicks'],
                'category_str': row['category_str'],
                'category': row['category'],
                'total_pageviews': row['total_pageviews'],
                'sentiment_label': row['sentiment_label'],
                'article_age': row['article_age']
            }
            expanded_rows.append(expanded_row)
    return pd.DataFrame(expanded_rows)

def augment_data(behaviors):
    """
    Augment the behavior data by duplicating clicked articles.

    """
    clicked = behaviors[behaviors['was_clicked'] == 1]
    non_clicked = behaviors[behaviors['was_clicked'] == 0]

    # Duplicate clicked articles
    augmented_clicked = clicked.sample(len(clicked) * 2, replace=True)
    augmented_behaviors = pd.concat([non_clicked, augmented_clicked])

    return augmented_behaviors

def oversample_data(features, target):
    """
    Oversample the training data using ADASYN.

    Parameters:
    features (DataFrame): Training features
    target (Series): Training target

    Returns:
    features_res DataFrame: Resampled training features
    target_res Series: Resampled training target
    """
    adasyn = ADASYN(random_state=42)
    features_res, target_res = adasyn.fit_resample(features, target)
    return features_res, target_res

def main():
    train_path = 'demo DataSet/train'
    val_path = 'demo DataSet/validation'

    articles = pd.read_parquet('demo DataSet/articles.parquet')
    behaviors_train, history_train = load_data(train_path)
    behaviors_val, history_val = load_data(val_path)

    train_behaviors = preprocess_data(behaviors_train, articles)
    val_behaviors = preprocess_data(behaviors_val, articles)

    train_behaviors = expand_and_label(train_behaviors)
    val_behaviors = expand_and_label(val_behaviors)

    # Augment data by duplicating clicked articles
    train_behaviors = augment_data(train_behaviors)

    features_train = train_behaviors.drop(columns=['was_clicked'])
    target_train = train_behaviors['was_clicked']
    features_val = val_behaviors.drop(columns=['was_clicked'])
    target_val = val_behaviors['was_clicked']

    # Oversample the training data using ADASYN
    features_train, target_train = oversample_data(features_train, target_train)

    model = train_model(features_train, target_train)
    print("Training Performance:")
    evaluate_model(model, features_train, target_train)
    print("Validation Performance:")
    evaluate_model(model, features_val, target_val)

    # Display feature importances
    feature_importances = model.feature_importances_
    sorted_idx = np.argsort(feature_importances)
    plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), [features_train.columns[i] for i in sorted_idx])
    plt.title('Feature Importance')
    plt.show()


if __name__ == "__main__":
    main()

