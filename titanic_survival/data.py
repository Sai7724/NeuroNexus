# titanic_app.py
from flask import Flask, render_template_string, request
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# ======================== DATA & MODEL ========================
def load_data():
    data = pd.read_csv('tested.csv')
    data['Title'] = data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    common_titles = ['Mr', 'Mrs', 'Miss', 'Master']
    data['Title'] = data['Title'].apply(lambda x: x if x in common_titles else 'Other')
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
    return data

def train_model(data):
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'IsAlone']
    X = data[features]
    y = data['Survived']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, ['Age', 'Fare']),
            ('cat', categorical_transformer, ['Pclass', 'Sex', 'Embarked', 'Title', 'IsAlone'])])
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, min_samples_leaf=2, max_features='sqrt'))],
        memory='cache_directory')
    
    model.fit(X, y)
    return model, data

data = load_data()
model, data = train_model(data)

# ======================== VISUALIZATIONS ========================
def create_plot(data, survival_status, plot_type):
    plt.figure(figsize=(8, 4))
    filtered = data[data['Survived'] == survival_status]
    
    if plot_type == 'age':
        filtered['Age'].hist(bins=10, color='green' if survival_status else 'red', edgecolor='black')
        plt.title('Age Distribution')
        plt.xlabel('Age')
    elif plot_type == 'gender':
        filtered['Sex'].value_counts().plot(kind='bar', color=['pink', 'lightblue'])
        plt.title('Gender Distribution')
        plt.xticks([0, 1], ['Female', 'Male'], rotation=0)
    elif plot_type == 'class':
        filtered['Pclass'].value_counts().sort_index().plot(kind='bar', color=['gold', 'silver', 'brown'])
        plt.title('Class Distribution')
        plt.xticks([0, 1, 2], ['1st', '2nd', '3rd'], rotation=0)
    
    plt.tight_layout()
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

# ======================== HTML TEMPLATES ========================
INDEX_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Titanic Survival Predictor</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding: 20px; background-color: #f8f9fa; }
        .form-container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        .header { margin-bottom: 30px; text-align: center; }
        .feature-card { margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚢 Titanic Survival Predictor</h1>
            <p class="lead">Would you have survived the Titanic disaster?</p>
        </div>
        
        <div class="row">
            <div class="col-md-8">
                <div class="form-container">
                    <form method="POST">
                        <div class="row g-3">
                            <!-- Passenger Details -->
                            <div class="col-md-6">
                                <label class="form-label">Passenger Class</label>
                                <select class="form-select" name="pclass" required>
                                    <option value="1">First Class</option>
                                    <option value="2">Second Class</option>
                                    <option value="3">Third Class</option>
                                </select>
                            </div>
                            
                            <div class="col-md-6">
                                <label class="form-label">Gender</label>
                                <select class="form-select" name="sex" required>
                                    <option value="female">Female</option>
                                    <option value="male">Male</option>
                                </select>
                            </div>
                            
                            <div class="col-md-6">
                                <label class="form-label">Age</label>
                                <input type="number" class="form-control" name="age" min="0" max="100" required>
                            </div>
                            
                            <div class="col-md-6">
                                <label class="form-label">Fare (£)</label>
                                <input type="number" step="0.01" class="form-control" name="fare" min="0" required>
                            </div>
                            
                            <div class="col-md-6">
                                <label class="form-label">Embarkation Port</label>
                                <select class="form-select" name="embarked" required>
                                    <option value="C">Cherbourg</option>
                                    <option value="Q">Queenstown</option>
                                    <option value="S">Southampton</option>
                                </select>
                            </div>
                            
                            <div class="col-md-6">
                                <label class="form-label">Title</label>
                                <select class="form-select" name="title" required>
                                    <option value="Mr">Mr</option>
                                    <option value="Mrs">Mrs</option>
                                    <option value="Miss">Miss</option>
                                    <option value="Master">Master</option>
                                    <option value="Other">Other</option>
                                </select>
                            </div>
                            
                            <div class="col-md-6">
                                <label class="form-label">Traveling Alone?</label>
                                <select class="form-select" name="is_alone" required>
                                    <option value="1">Yes</option>
                                    <option value="0">No</option>
                                </select>
                            </div>
                            
                            <div class="col-md-6">
                                <label class="form-label">Family Size</label>
                                <input type="number" class="form-control" name="family_size" min="1" max="10" value="1">
                            </div>
                        </div>
                        
                        <div class="d-grid gap-2 mt-4">
                            <button class="btn btn-primary btn-lg" type="submit">Predict Survival</button>
                        </div>
                    </form>
                </div>
            </div>
            
            <div class="col-md-4">
                <div class="card feature-card">
                    <div class="card-body">
                        <h5 class="card-title">📊 Survival Rates by Class</h5>
                        <img src="data:image/png;base64,{{ class_plot }}" class="img-fluid">
                    </div>
                </div>
                
                <div class="card feature-card">
                    <div class="card-body">
                        <h5 class="card-title">👫 Gender Survival Comparison</h5>
                        <img src="data:image/png;base64,{{ gender_plot }}" class="img-fluid">
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
'''

RESULT_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Prediction Result</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding: 20px; background-color: #f8f9fa; }
        .result-container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        .prediction { font-size: 2rem; font-weight: bold; margin: 20px 0; text-align: center; }
        .survived { color: green; }
        .not-survived { color: red; }
        .plot-container { margin: 30px 0; }
        .plot { max-width: 100%; height: auto; }
        .stats-card { margin-bottom: 20px; }
        .similar-passengers { max-height: 300px; overflow-y: auto; }
    </style>
</head>
<body>
    <div class="container">
        <div class="result-container">
            <h1>Prediction Result</h1>
            
            <div class="prediction {% if result == 'Survived' %}survived{% else %}not-survived{% endif %}">
                {% if result == 'Survived' %}🎉{% else %}💀{% endif %}
                You would have <strong>{{ result }}</strong> ({{ probability }}% chance)
                {% if result == 'Survived' %}🎉{% else %}💀{% endif %}
            </div>
            
            <div class="row">
                <div class="col-md-6">
                    <div class="card stats-card">
                        <div class="card-body">
                            <h5 class="card-title">📊 Your Statistics</h5>
                            <ul class="list-group list-group-flush">
                                <li class="list-group-item">Class: {{ "1st" if pclass == 1 else "2nd" if pclass == 2 else "3rd" }}</li>
                                <li class="list-group-item">Gender: {{ sex|title }}</li>
                                <li class="list-group-item">Age: {{ age }}</li>
                                <li class="list-group-item">Fare: £{{ fare|round(2) }}</li>
                                <li class="list-group-item">Family Size: {{ family_size }}</li>
                            </ul>
                        </div>
                    </div>
                    
                    <div class="card stats-card">
                        <div class="card-body">
                            <h5 class="card-title">📈 Survival Probability</h5>
                            <div class="progress" style="height: 30px;">
                                <div class="progress-bar {% if result == 'Survived' %}bg-success{% else %}bg-danger{% endif %}" 
                                     role="progressbar" style="width: {{ probability }}%;" 
                                     aria-valuenow="{{ probability }}" aria-valuemin="0" aria-valuemax="100">
                                    {{ probability }}%
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="col-md-6">
                    <div class="card stats-card">
                        <div class="card-body">
                            <h5 class="card-title">👥 Similar Passengers ({{ similar_count }})</h5>
                            <div class="similar-passengers">
                                <table class="table">
                                    <thead>
                                        <tr>
                                            <th>Class</th>
                                            <th>Age</th>
                                            <th>Gender</th>
                                            <th>Survived</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {% for passenger in similar_passengers %}
                                        <tr>
                                            <td>{{ passenger.Pclass }}</td>
                                            <td>{{ passenger.Age|round(1) }}</td>
                                            <td>{{ passenger.Sex|title }}</td>
                                            <td>{% if passenger.Survived == 1 %}✅{% else %}❌{% endif %}</td>
                                        </tr>
                                        {% endfor %}
                                    </tbody>
                                </table>
                            </div>
                            <p class="mt-2">Historical survival rate: <strong>{{ (survival_rate * 100)|round(1) }}%</strong></p>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="plot-container">
                <h3>Demographic Analysis</h3>
                <div class="row">
                    <div class="col-md-4">
                        <h5>Age Distribution</h5>
                        <img src="data:image/png;base64,{{ age_plot }}" class="plot">
                    </div>
                    <div class="col-md-4">
                        <h5>Gender Distribution</h5>
                        <img src="data:image/png;base64,{{ gender_plot }}" class="plot">
                    </div>
                    <div class="col-md-4">
                        <h5>Class Distribution</h5>
                        <img src="data:image/png;base64,{{ class_plot }}" class="plot">
                    </div>
                </div>
            </div>
            
            <div class="d-grid gap-2">
                <a href="/" class="btn btn-primary">Make Another Prediction</a>
            </div>
        </div>
    </div>
</body>
</html>
'''

# ======================== FLASK ROUTES ========================
@app.route('/', methods=['GET'])
def index():
    # Generate initial plots
    class_plot = create_plot(data, 1, 'class')
    gender_plot = create_plot(data, 1, 'gender')
    return render_template_string(INDEX_HTML, class_plot=class_plot, gender_plot=gender_plot)

@app.route('/', methods=['POST'])
def predict():
    # Get form data
    pclass = int(request.form['pclass'])
    sex = request.form['sex']
    age = float(request.form['age'])
    fare = float(request.form['fare'])
    embarked = request.form['embarked']
    title = request.form['title']
    is_alone = int(request.form['is_alone'])
    family_size = int(request.form.get('family_size', 1))
    
    # Make prediction
    passenger = pd.DataFrame([{
        'Pclass': pclass, 'Sex': sex, 'Age': age, 'Fare': fare,
        'Embarked': embarked, 'Title': title, 'IsAlone': is_alone
    }])
    
    # Get probability
    proba = model.predict_proba(passenger)[0][1]
    result = "Survived" if proba >= 0.5 else "Did Not Survive"
    probability = round(proba * 100, 1) if result == "Survived" else round((1 - proba) * 100, 1)
    
    # Get similar passengers
    similar = data[
        (data['Pclass'] == pclass) & 
        (data['Sex'] == sex) &
        (data['Age'].between(age-5, age+5))
    ].sample(min(5, len(data)), random_state=42)
    similar_count = len(similar)
    survival_rate = similar['Survived'].mean() if not similar.empty else 0
    
    # Generate all plots
    age_plot = create_plot(data, 1 if result == "Survived" else 0, 'age')
    gender_plot = create_plot(data, 1 if result == "Survived" else 0, 'gender')
    class_plot = create_plot(data, 1 if result == "Survived" else 0, 'class')
    
    return render_template_string(
        RESULT_HTML,
        result=result,
        probability=probability,
        pclass=pclass,
        sex=sex,
        age=age,
        fare=fare,
        family_size=family_size,
        similar_count=similar_count,
        survival_rate=survival_rate,
        similar_passengers=similar.to_dict('records'),
        age_plot=age_plot,
        gender_plot=gender_plot,
        class_plot=class_plot
    )

if __name__ == '__main__':
    app.run(debug=True)