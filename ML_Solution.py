import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    # Load data
    behaviors = pd.read_parquet(f'{path}/behaviors.parquet')
    history = pd.read_parquet(f'{path}/history.parquet')
    print(behaviors.columns)
    print(history.columns)
    return behaviors, history


def preprocess_data(behaviors, history, articles):

    # Convert 'time' to datetime and extract hour and day of the week
    behaviors['hour_of_day'] = pd.to_datetime(behaviors['impression_time']).dt.hour
    behaviors['day_of_week'] = pd.to_datetime(behaviors['impression_time']).dt.dayofweek

    # Select specific features from behaviors
    behaviors = behaviors[['user_id', 'article_id', 'article_ids_inview', 'article_ids_clicked', 'hour_of_day',
         'day_of_week', 'device_type', 'is_subscriber', 'read_time', 'scroll_percentage']]


    # Convert 'time_published' and 'time_modified' to year
    articles['year_published'] = pd.to_datetime(articles['published_time']).dt.year
    articles['year_modified'] = pd.to_datetime(articles['published_time']).dt.year

    # Select specific features and apply label encoding
    label_encoder = LabelEncoder()
    articles['category_str'] = label_encoder.fit_transform(articles['category_str'])
    articles['article_type'] = label_encoder.fit_transform(articles['article_type'])
    articles['sentiment_label'] = label_encoder.fit_transform(articles['sentiment_label'])

    # Select specific features
    articles = articles[
        ['article_id', 'category_str', 'category', 'year_published', 'year_modified', 'article_type',
         'total_pageviews', 'sentiment_label']]

    # Merge article data to behaviors
    behaviors = behaviors.merge(articles, on='article_id', how='left')


    return behaviors


def train_model(features, target):
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, class_weight='balanced')
    # scores = cross_val_score(model, features, target, cv=5)
    # print("Cross-validated accuracy scores:", scores)
    # print("Average cross-validation score:", np.mean(scores))
    model.fit(features, target)
    return model


def evaluate_model(model, features, target):
    predictions = model.predict(features)
    prediction_probs = model.predict_proba(features)[:,
                       1]  # assuming binary classification and you want the probability of the positive class

    acc = accuracy_score(target, predictions)
    print("Accuracy:", acc)

    # Detailed classification report
    class_report = classification_report(target, predictions)
    print("Classification Report:")
    print(class_report)

    # Calculating ROC AUC Score
    roc_auc = roc_auc_score(target, prediction_probs)
    print("ROC AUC Score:", roc_auc)

    return acc, class_report, roc_auc

def expand_and_label(behaviors):
    # Creating a new DataFrame to hold the expanded rows
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
                # Add article-specific features
                'category_str': row['category_str'],
                'category': row['category'],
                'year_published': row['year_published'],
                'year_modified': row['year_modified'],
                'article_type': row['article_type'],
                'total_pageviews': row['total_pageviews'],
                'sentiment_label': row['sentiment_label']
            }
            expanded_rows.append(expanded_row)
    return pd.DataFrame(expanded_rows)


def main():
    # Load and preprocess data as before
    train_path = 'demo DataSet/train'
    val_path = 'demo DataSet/validation'

    articles = pd.read_parquet('demo DataSet/articles.parquet')
    behaviors_train, history_train = load_data(train_path)
    behaviors_val, history_val = load_data(val_path)

    # Preprocess data
    train_behaviors = preprocess_data(behaviors_train, history_train, articles)
    val_behaviors = preprocess_data(behaviors_val, history_val, articles)

    # Expand and label data
    train_behaviors = expand_and_label(train_behaviors)
    val_behaviors = expand_and_label(val_behaviors)



    # Extract features and target
    features_train = train_behaviors.drop(columns=['was_clicked'])
    target_train = train_behaviors['was_clicked']
    features_val = val_behaviors.drop(columns=['was_clicked'])
    target_val = val_behaviors['was_clicked']

    # Train model
    model = train_model(features_train, target_train)
    print("Training Performance:")
    evaluate_model(model, features_train, target_train)
    print("Validation Performance:")
    evaluate_model(model, features_val, target_val)


if __name__ == "__main__":
    main()
