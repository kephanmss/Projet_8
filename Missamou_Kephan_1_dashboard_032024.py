import os
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import mlflow
import mlflow.sklearn
import shutil
from typing import Type
from pydantic import BaseModel, create_model
import streamlit as st
import pandas as pd
import numpy as np
import shap
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from googletrans import Translator
import asyncio
from dotenv import load_dotenv

def get_s3_file(bucket_name, s3_path, local_path):
    """
    Télécharge un fichier depuis S3 et le sauvegarde localement.
    """
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'https://s3.amazonaws.com'
    load_dotenv()
    # Access AWS credentials from environment variables
    os.environ['AWS_ACCESS_KEY_ID'] = os.environ.get('ACCESS_KEY')
    os.environ['AWS_SECRET_ACCESS_KEY'] = os.environ.get('SECRET_KEY')
    os.environ['AWS_DEFAULT_REGION'] = os.environ.get('AWS_DEFAULT_REGION', 'eu-north-1')
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
        region_name=os.environ['AWS_DEFAULT_REGION']
    )

    if not os.path.exists(os.path.dirname(local_path)):
        os.makedirs(os.path.dirname(local_path))
    try:
        # Download the file from S3
        filename = os.path.basename(s3_path)
        for file in s3_client.list_objects_v2(Bucket=bucket_name, Prefix=s3_path).get('Contents', []):
            if file['Key'] == s3_path:
                try:
                    s3_client.download_file(bucket_name, file['Key'], local_path)
                    print(f"Fichier téléchargé avec succès depuis s3://{bucket_name}/{s3_path} vers {local_path}")
                except ClientError as e:
                    print(f"Erreur lors du téléchargement du fichier: {e}")
    except NoCredentialsError:
        print("Erreur: Les identifiants AWS ne sont pas disponibles.")
    except ClientError as e:
        print(f"Erreur lors de l'accès à S3: {e}")
        if "AccessDenied" in str(e):
            print("Accès au bucket refusé. Vérifiez vos permissions.")
            print("Suggestions:")
            print("1. Vérifiez que le nom du bucket est correct")
            print("2. Vérifiez que vous avez les permissions suivantes:")
            print("   - s3:GetObject")
            print("   - s3:ListBucket")
            print("3. Vérifiez que vos clés d'accès AWS sont valides")
            print("4. Vérifiez que la région est correcte")

def get_s3_csv(bucket_name, s3_path, local_path, encoding, sep):
        """
        Télécharge un fichier CSV depuis S3 et le sauvegarde localement.
        """
        try:
            # Download the file from S3
            get_s3_file(bucket_name, s3_path, local_path)
            print(f"Fichier téléchargé avec succès depuis s3://{bucket_name}/{s3_path} vers {local_path}")
            # Read the CSV file
            df = pd.read_csv(local_path, sep=sep, encoding=encoding)
            return df
        except NoCredentialsError:
            print("Erreur: Les identifiants AWS ne sont pas disponibles.")
        except ClientError as e:
            print(f"Erreur lors de l'accès à S3: {e}")
            if "AccessDenied" in str(e):
                print("Accès au bucket refusé. Vérifiez vos permissions.")
                print("Suggestions:")
                print("1. Vérifiez que le nom du bucket est correct")
                print("2. Vérifiez que vous avez les permissions suivantes:")
                print("   - s3:GetObject")
                print("   - s3:ListBucket")
                print("3. Vérifiez que vos clés d'accès AWS sont valides")
                print("4. Vérifiez que la région est correcte")

class Utils:

    def __init__(self):
        pass


    @staticmethod
    def case_when(*args):
        """
        Fonction pour créer une colonne avec des conditions.
        """
        if len(args) % 2 != 0:
            raise ValueError("Le nombre d'arguments doit être pair.")
        conditions = args[::2]
        values = args[1::2]
        return np.select(conditions, values, default=None)

    @staticmethod
    def load_model_from_s3():
        """"
        Va chercher le modèle depuis le bucket S3, le télécharge et le charge.
        """
        load_dotenv()
        # Configure AWS S3 settings
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'https://s3.amazonaws.com'
        # Load environment variables from .env file
        load_dotenv()
        
        # Access AWS credentials from environment variables
        os.environ['AWS_ACCESS_KEY_ID'] = os.environ.get('ACCESS_KEY')
        os.environ['AWS_SECRET_ACCESS_KEY'] = os.environ.get('SECRET_KEY')
        os.environ['AWS_DEFAULT_REGION'] = os.environ.get('AWS_DEFAULT_REGION', 'eu-north-1')

        # Define S3 bucket and key
        bucket_name = 'projet-7-opc'
        s3_prefix = 'models/my-model'
        local_model_path = './Data/Modele/'

        # modele
        if st.session_state.get('model') == None:

            try:
                # Initialize S3 client
                s3_client = boto3.client(
                    's3',
                    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
                    region_name=os.environ['AWS_DEFAULT_REGION']
                )
                
                # Test access to the bucket
                print(f"Test de l'accès au bucket: {bucket_name}")
                try:
                    s3_client.list_objects_v2(Bucket=bucket_name, MaxKeys=1)
                    print(f"Accès au bucket: {bucket_name} confirmé")
                except ClientError as e:
                    print(f"Erreur lors de la tentative d'accès au bucket {bucket_name}: {e}")
                    print("Vérifiez que le nom du bucket est correct et que vous avez les permissions nécessaires.")
                    raise
                
                # Create a local directory to store the downloaded model
                if os.path.exists(local_model_path):
                    shutil.rmtree(local_model_path)  # Remove if exists
                os.makedirs(local_model_path)
                
                # List all objects under the s3_prefix
                response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=s3_prefix)
                
                if 'Contents' not in response:
                    print(f"Rien trouvé dans s3://{bucket_name}/{s3_prefix}")
                    exit(1)
                
                # Download each file from S3
                for item in response['Contents']:
                    s3_key = item['Key']
                    # Create local path maintaining folder structure
                    relative_path = s3_key.replace(s3_prefix + '/', '')
                    local_file_path = os.path.join(local_model_path, relative_path)
                    
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                    
                    print(f"Téléchargement de s3://{bucket_name}/{s3_key} vers {local_file_path}")
                    s3_client.download_file(bucket_name, s3_key, local_file_path)
                
                print(f"Modèle téléchargé avec succès dans {local_model_path}")
                
                # Load the model
                model = mlflow.sklearn.load_model(local_model_path)
                st.session_state.model = model

                if st.session_state.get('model') is None:
                    raise ValueError("Le modèle n'a pas pu être chargé.")
                
                print("Model loaded successfully!")
                
            except NoCredentialsError:
                print("AWS credentials not found or invalid")
            except ClientError as e:
                print(f"AWS S3 error: {e}")
                if "AccessDenied" in str(e):
                    print("Accès au bucket refusé. Vérifiez vos permissions.")
                    print("Suggestions:")
                    print("1. Vérifiez que le nom du bucket est correct")
                    print("2. Vérifiez que vous avez les permissions suivantes:")
                    print("   - s3:GetObject")
                    print("   - s3:ListBucket")
                    print("3. Vérifiez que vos clés d'accès AWS sont valides")
                    print("4. Vérifiez que la région est correcte")
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("Utilisation du modèle mise en cache.")
        return None

    @staticmethod
    def create_feature_model(path: str = './Data/Prep/feature_names.csv', sep:str = ';', encoding:str = 'utf-8') -> Type[BaseModel]:
        df = get_s3_csv(
            bucket_name='projet-7-opc',
            s3_path='data/Data/Prep/feature_names.csv',
            local_path=path,
            encoding=encoding,
            sep=sep
        )
        df = df.dropna()
        fields = {}
        for _, row in df.iterrows():
            feature_name = row['features']
            feature_type = row['type']
            if feature_type.startswith('int'):
                fields[feature_name] = (int, ...)
            elif feature_type.startswith('float'):
                fields[feature_name] = (float, ...)
            else:
                fields[feature_name] = (str, ...)
        feature_model = create_model('FeatureModel', **fields)
        st.session_state.feature_model = feature_model
        return None

    @staticmethod
    def init_feature_names(path: str = './Data/Prep/feature_names.csv', sep:str = ';', encoding:str = 'utf-8') -> list:
        df = get_s3_csv(
            bucket_name='projet-7-opc',
            s3_path='data/Data/Prep/feature_names.csv',
            local_path=path,
            encoding=encoding,
            sep=sep
        )
        df = df.dropna()
        feature_names = df['features'].tolist()
        st.session_state.feature_names = feature_names
        return None

    @staticmethod
    def get_columns_descriptions(path: str = './Data/Prep/HomeCredit_columns_description.csv', sep:str = ',', encoding:str = 'utf-8') -> dict:
        """
        Charge les descriptions des colonnes depuis un fichier CSV.
        """
        df = get_s3_csv(
            bucket_name='projet-7-opc',
            s3_path='data/Data/Prep/HomeCredit_columns_description.csv',
            local_path=path,
            encoding=encoding,
            sep=sep
        )
        df = df.loc[:, ['Row', 'Description']]
        df.rename(columns={'Row': 'Features', 'Description': 'Description'}, inplace=True)
        st.session_state.columns_descriptions = df
        return None

    @staticmethod
    def init_shap_explainer(model):
        """
        Initialise l'explainer SHAP pour le modèle donné.
        """
        if model is None:
            raise ValueError("Le modèle est None. Veuillez charger un modèle valide.")
        
        # On utilise le modèle pour créer un explainer SHAP
        explainer = shap.Explainer(model)
        tree_explainer = shap.TreeExplainer(model)
        st.session_state.explainer = explainer
        st.session_state.tree_explainer = tree_explainer
        return None
    
    @staticmethod
    def get_shap_values(explainer, data, n_sample:int = 1000):
        """
        Calcule les valeurs SHAP pour un échantillon de données donné.
        """
        if explainer is None:
            raise ValueError("L'explainer SHAP est None. Veuillez initialiser l'explainer.")
        
        with st.spinner("Calcul des valeurs SHAP...", show_time=True):
            # On utilise l'explainer pour calculer les valeurs SHAP
            shap_values = explainer(data.sample(n=n_sample, random_state=42))
            st.session_state.shap_values = shap_values
        return None

    @staticmethod
    def load_data(feature_list: list = None):
        """
        Charge les données

        Args:
            feature_list (list): Liste des features à charger.
        """
        root_path = 'data/Data/Input_data/'
        train = [
            'X_train.csv',
            'train_ids.csv',
            'y_train.csv',
        ]

        test = [
            'X_test.csv',
            'test_ids.csv',
            'y_test.csv',
        ]
        download_root_path = './Data/Input_data/'
        if not os.path.exists(download_root_path):
            os.makedirs(download_root_path)

        # On charge les données d'entrainement
        train_data = get_s3_csv(
            bucket_name='projet-7-opc',
            s3_path='data/Data/Input_data/X_train.csv',
            local_path='./Data/Input_data/X_train.csv',
            encoding='utf-8',
            sep=';'
        )

        train_ids = get_s3_csv(
            bucket_name='projet-7-opc',
            s3_path=os.path.join(root_path, train[1]),
            local_path=os.path.join(download_root_path, train[1]),
            encoding='utf-8',
            sep=';'
        )
        train_data['SK_ID_CURR'] = train_ids['SK_ID_CURR']

        # On charge les données de test
        test_data = get_s3_csv(
            bucket_name='projet-7-opc',
            s3_path=os.path.join(root_path, test[0]),
            local_path=os.path.join(download_root_path, test[0]),
            encoding='utf-8',
            sep=';'
        )
        test_ids = get_s3_csv(
            bucket_name='projet-7-opc',
            s3_path=os.path.join(root_path, test[1]),
            local_path=os.path.join(download_root_path, test[1]),
            encoding='utf-8',
            sep=';'
        )
        test_data['SK_ID_CURR'] = test_ids['SK_ID_CURR']

        # Concatène les données d'entrainement et de test
        whole_data = pd.concat([train_data, test_data], axis=0, ignore_index=True)

        if feature_list is not None:
            # On ne garde que les features demandées
            whole_data = whole_data[feature_list + ['SK_ID_CURR']]

        st.session_state.data = whole_data
        st.session_state.all_ids = whole_data['SK_ID_CURR'].tolist()
        return None

    @staticmethod
    def get_raw_data():
        """
        Retourne les données brutes.
        """
        paths = [
            'data/Data/Input_data/application_train.csv',
            'data/Data/Input_data/application_test.csv'
        ]

        data = pd.DataFrame()
        for path in paths:
            df = get_s3_csv(
                bucket_name='projet-7-opc',
                s3_path=path,
                local_path='./Data/Input_data/' + os.path.basename(path),
                encoding='utf-8',
                sep=','
            )
            data = pd.concat([data, df], axis=0, ignore_index=True)
        data = data.loc[data['SK_ID_CURR'].isin(st.session_state.all_ids), :]
        st.session_state.raw_data = data
        return None

    @staticmethod
    def get_model_features(model):
        """
        Retourne les features du modèle donné.
        """
        if model is None:
            raise ValueError("Le modèle est None. Veuillez charger un modèle valide.")
        
        # On utilise le modèle pour obtenir les features
        feature_names = model.get_feature_names_out()
        st.session_state.feature_names = feature_names
        return None

    @staticmethod
    def run_model(model, data):
        """
        Exécute le modèle sur les données fournies et retourne les prédictions ainsi que leur probabilité.
        """
        if model is None:
            raise ValueError("Le modèle est None. Veuillez charger un modèle valide.")
        
        if 'SK_ID_CURR' in data.columns:
            data = data.drop(columns=['SK_ID_CURR'])
        
        # On utilise le modèle pour faire des prédictions
        predictions = model.predict(data)
        probabilities = model.predict_proba(data)[:, 1]
        st.session_state.predictions = predictions
        st.session_state.probabilities = probabilities
        return None

    @staticmethod
    def load_pretraitement():
        """
        Charge le scaler pour les données.
        """
        # Get joblib files from S3
        paths = [
            'data/Data/Prep/cat_imputer.joblib',
            'data/Data/Prep/num_imputer.joblib',
            'data/Data/Prep/scaler.joblib',
            'data/Data/Prep/onehot.joblib'
        ]

        for path in paths:
            get_s3_file(
                bucket_name='projet-7-opc',
                s3_path=path,
                local_path='./Data/Prep/' + os.path.basename(path)
            )

        cat_imputer = joblib.load('./Data/Prep/cat_imputer.joblib')
        num_imputer = joblib.load('./Data/Prep/num_imputer.joblib')
        scaler = joblib.load('./Data/Prep/scaler.joblib')
        onehot = joblib.load('./Data/Prep/onehot.joblib')
        st.session_state.cat_imputer = cat_imputer
        st.session_state.num_imputer = num_imputer
        st.session_state.scaler = scaler
        st.session_state.onehot = onehot
        return None

def init_session_state_variables():
    """
    Initialise les variables de session pour Streamlit.
    """
    if st.session_state.get('init_done') is None or not st.session_state.get('init_done'):
        st.title("Initialisation des variables de session")
        st.write("Veuillez patienter et ne pas recharger la page. Cette opération ne se fait qu'une seule fois.")
        progress_bar = st.progress(0, text="Initialisation en cours...")
        step_number = 7
        # 1 - Modèle S3
        if st.session_state.get('model') is None:
            Utils.load_model_from_s3()
            progress_bar.progress(
                (1 / step_number)
            )
        # Features
            # 2 - FeatureModel
        if st.session_state.get('feature_model') is None:
            Utils.create_feature_model()
            progress_bar.progress(
                (2 / step_number)
            )
            # 3 - Feature names
        if st.session_state.get('feature_names') is None:
            Utils.init_feature_names()
            progress_bar.progress(
                (3 / step_number)
            )
            # 4 - Descriptions des colonnes
        if st.session_state.get('columns_descriptions') is None:
            Utils.get_columns_descriptions()
            progress_bar.progress(
                (4 / step_number)
            )
        # 5 - Données
        if st.session_state.get('data') is None:
            Utils.load_data(
                feature_list=st.session_state.get('feature_names')
            )
            Utils.get_raw_data()
            Utils.load_pretraitement()
            progress_bar.progress(
                (5 / step_number)
            )
        # 6 - Prédictions/Probabilités
        if (st.session_state.get('predictions') is None) or (st.session_state.get('probabilities') is None):
            Utils.run_model(
                model=st.session_state.get('model'), 
                data=st.session_state.get('data')
            )
            progress_bar.progress(
                (6 / step_number)
            )
        # 7 - SHAP values
        if st.session_state.get('shap_values') is None:
            Utils.init_shap_explainer(st.session_state.get('model'))
            progress_bar.progress(
                (7 / step_number)
            )
        st.success("Initialisation terminée ! Vous pouvez maintenant utiliser le tableau de bord.")
        st.session_state['init_done'] = True
    return None

def prediction():

    st.title("Prédiction")
    # Afficher la classe prédite
    st.subheader(f"Classe prédite pour le client n°{st.session_state.id_client}")
    predictions = st.session_state.get('predictions')[st.session_state.index_client]
    probabilites = st.session_state.get('probabilities')[st.session_state.index_client]
    client_prediction = int(predictions)
    client_probability = float(probabilites)
    solvabiliy_rate = 100 * client_probability
    cutoff_threshold = 0.5
    st.metric(
        label="Probabilité de solvabilité",
        value=f"{solvabiliy_rate:.2f}%",
        delta=f"{100*(client_probability - cutoff_threshold):.2f}%",
        delta_color="normal",
        help="Probabilité que le client soit solvable, ie qu'il rembourse son prêt, déterminée par le modèle de Machine Learning. \
            La probabilité est calculée en fonction des caractéristiques du client et de l'algorithme de Machine Learning utilisé. \
                **La flèche indique si la probabilité est supérieure ou inférieure au seuil de décision, fixé à 50% par défaut. \
                Si la probabilité est supérieure au seuil, le client est considéré comme solvable.**"
    )
    human_readable_prediction = "Non solvable" if client_prediction == 0 else "Solvable"
    if human_readable_prediction == "Solvable":
        st.success(f"Le modèle prédit que le client n°{st.session_state.id_client} est **{human_readable_prediction}**.")
    else:
        st.error(f"Le modèle prédit que le client n°{st.session_state.id_client} est **{human_readable_prediction}**.")


    # Afficher les shap values principales expliquant la prédiction
    st.subheader(f"Raisons du score pour le client n°{st.session_state.id_client}", 
                 help="Valeurs SHAP pour le client sélectionné. \
                    Les valeurs SHAP indiquent l'importance de chaque caractéristique client dans la prédiction du modèle. \
                    Les valeurs positives indiquent que la caractéristique contribue à augmenter la probabilité de solvabilité, \
                    tandis que les valeurs négatives indiquent une contribution à diminuer cette probabilité. \
                    Cette contribution (positive comme négative) est d'autant plus marquée que la valeur est élevée.")
    st.radio(
        label="Mode d'affichage des valeurs SHAP",
        options=['Réduit', 'Complet'],
        index=0,
        key='shap_mode',
        on_change=lambda: st.session_state.update({'mode': st.session_state.shap_mode.lower()}),
        help="Mode d'affichage des valeurs SHAP. \
        Le mode 'Réduit' affiche uniquement les caractéristiques les plus importantes, \
        tandis que le mode 'Complet' affiche toutes les caractéristiques."
    )
    explainer = st.session_state.get('explainer')
    client_data_for_shap = st.session_state.get('data').drop(columns=['SK_ID_CURR']).iloc[st.session_state.index_client:st.session_state.index_client+1]
    
    individual_shap_values = explainer(client_data_for_shap)
    
    def get_shap_df(values = individual_shap_values, mode:str = 'réduit'):
        """
        Fonction pour obtenir le DataFrame des valeurs SHAP.
        """

        # We select shap values for class 1: individual_shap_values.values[0, :, 1]
        shap_values_for_class_1 = individual_shap_values.values[0, :, 1]
        shap_df = pd.DataFrame({
            'Feature': client_data_for_shap.columns,
            'SHAP Value': shap_values_for_class_1
        })
        shap_df = shap_df.sort_values(by='SHAP Value', ascending=False)

        if mode == 'réduit':
            # Garder les features qui portent sont dans le top 75% (en valeur absolue, et en termes de quantiles) des valeurs SHAP
            q95 = shap_df['SHAP Value'].abs().quantile(0.95)
            shap_df['To display'] = Utils.case_when(
                shap_df['SHAP Value'].abs() > q95, True,
                shap_df['SHAP Value'].abs() < q95, False
            )
            new_df = pd.DataFrame({
                'Feature' : ['Autres impacts positifs', 'Autres impacts négatifs'],
                'SHAP Value' : [0, 0]
            })
            # On garde les features qui portent sont dans le top 75% (en valeur absolue, et en termes de quantiles) des valeurs SHAP
            # Le reste est mis dans des "Autres impacts positifs" ou "Autres impacts négatifs" condensés
            for i, row in shap_df.iterrows():
                if row['To display']:
                    new_df = pd.concat([new_df, row.to_frame().T], ignore_index=True)
                else:
                    if row['SHAP Value'] > 0:
                        new_df.loc[new_df['Feature'] == 'Autres impacts positifs', 'SHAP Value'] += row['SHAP Value']
                    else:
                        new_df.loc[new_df['Feature'] == 'Autres impacts négatifs', 'SHAP Value'] += row['SHAP Value']
        
        if mode == 'complet':
            # Garder toutes les features
            new_df = shap_df.copy()

        # Rajouter la colonne de valeur absolue si pas déjà présente
        new_df['Abs SHAP Value'] = new_df['SHAP Value'].abs()
        # Trier le DataFrame par valeur absolue décroissante
        new_df = new_df.sort_values(by='Abs SHAP Value', ascending=False)
        # Réinitialiser l'index
        new_df.reset_index(drop=True, inplace=True)
        return new_df

    # Initialiser le mode
    if 'mode' not in st.session_state:
        st.session_state.mode = 'réduit'

    def get_chart(shap_df):
        chart = alt.Chart(get_shap_df(mode=st.session_state.mode)).mark_bar().encode(
            x='SHAP Value', # Valeur SHAP
            y=alt.Y('Feature', sort=alt.EncodingSortField(field="Abs SHAP Value", op="sum", order='descending')), # Nom de la feature + tri par valeur SHAP décroissant
            color=alt.condition(
                alt.datum['SHAP Value'] > 0, # Condition pour la couleur
                alt.value('green'),
                alt.value('red')
            ), # Couleur en fonction de la valeur SHAP : bleu si positive, rouge si négative
            tooltip=['Feature', 'SHAP Value'] # Info-bulle avec le nom de la feature et la valeur SHAP
        ).properties(
            title='Importance des caractéristiques (Valeurs SHAP)' # Titre du graphique
        )
        return chart
    st.altair_chart(get_chart(get_shap_df(mode=st.session_state.mode)), 
                    use_container_width=True)

def positionnement():
    st.title("Positionnement du client par rapport à la population")
    
    nb_clients = st.session_state.get('data').shape[0]
    nb_clients_solvables = np.sum(st.session_state.get('predictions'))
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="Nombre total de clients dans la population",
            value=nb_clients,
            delta_color="normal",
            help="Nombre total de clients dans la population. \
                Cette valeur est calculée en comptant le nombre de lignes dans le DataFrame des données."
        )
    with col2:
        st.metric(
            label="Part de clients prédits solvables dans la population",
            value=f"{(nb_clients_solvables / nb_clients) * 100:.1f}%",
            delta_color="normal",
            help="Part de clients prédits solvables dans la population. \
                Cette valeur est calculée en divisant le nombre de clients solvables par le nombre total de clients dans la population."
        )
    client_prediction = st.session_state.get('predictions')[st.session_state.index_client]
    human_readable_prediction = "Non solvable" if client_prediction == 0 else "Solvable"
    if human_readable_prediction == "Solvable":
        st.success(f"Le modèle prédit que le client n°{st.session_state.id_client} est **{human_readable_prediction}**.")
    else:
        st.error(f"Le modèle prédit que le client n°{st.session_state.id_client} est **{human_readable_prediction}**.")

    if not st.session_state.get('init_done') or not st.session_state.get('id_client'):
        st.warning("Les données ne sont pas initialisées ou aucun client n'est sélectionné. Veuillez attendre la fin de l'initialisation et sélectionner un client dans la barre latérale.")
        return
    
    explainer = st.session_state.get('explainer')
    client_data_for_shap = st.session_state.get('data').drop(columns=['SK_ID_CURR']).iloc[st.session_state.index_client:st.session_state.index_client+1]
    
    individual_shap_values = explainer(client_data_for_shap)
    
    def get_shap_df(values = individual_shap_values, mode:str = 'réduit'):
        """
        Fonction pour obtenir le DataFrame des valeurs SHAP.
        """

        # We select shap values for class 1: individual_shap_values.values[0, :, 1]
        shap_values_for_class_1 = individual_shap_values.values[0, :, 1]
        shap_df = pd.DataFrame({
            'Feature': client_data_for_shap.columns,
            'SHAP Value': shap_values_for_class_1
        })
        shap_df['Abs SHAP Value'] = shap_df['SHAP Value'].abs()
        shap_df = shap_df.sort_values(by='Abs SHAP Value', ascending=False)
        shap_df.drop(columns=['Abs SHAP Value'], inplace=True)
        st.session_state['top_shap_features'] = shap_df
        return None
    # On appelle la fonction pour trier les features par valeur SHAP décroissante
    with st.spinner("Calcul des valeurs SHAP pour la population...", show_time=True):
        get_shap_df()

    def reverse_pretreatment(data):
        """
        Fonction pour dénormaliser les données.
        """
        # On inverse le prétraitement des données
        cat_imputer = st.session_state.get('cat_imputer')
        num_imputer = st.session_state.get('num_imputer')
        scaler = st.session_state.get('scaler')
        onehot = st.session_state.get('onehot')

        if scaler is not None:
            # Get all feature names the scaler was fitted on
            all_fitted_scaler_features = scaler.get_feature_names_out()
            # Identify features that are in the current data AND were handled by this scaler
            features_to_transform = [col for col in all_fitted_scaler_features if col in data.columns]
            
            if features_to_transform:
                # Perform manual inverse transformation for each relevant column
                for col_name in features_to_transform:
                    try:
                        # Find the index of the current column in the scaler's list of all fitted features
                        idx_in_scaler = list(all_fitted_scaler_features).index(col_name)
                        
                        # Apply inverse transformation: X_original = (X_scaled * scale) + mean
                        # Ensure scaler.scale_ and scaler.mean_ are accessed with the correct index
                        if hasattr(scaler, 'scale_') and scaler.scale_ is not None and \
                           hasattr(scaler, 'mean_') and scaler.mean_ is not None and \
                           hasattr(scaler, 'with_std') and scaler.with_std and \
                           hasattr(scaler, 'with_mean') and scaler.with_mean:
                            data[col_name] = data[col_name] * scaler.scale_[idx_in_scaler] + scaler.mean_[idx_in_scaler]
                        elif hasattr(scaler, 'scale_') and scaler.scale_ is not None and \
                             hasattr(scaler, 'with_std') and scaler.with_std: # and not scaler.with_mean (or mean_ is None)
                            data[col_name] = data[col_name] * scaler.scale_[idx_in_scaler]
                        elif hasattr(scaler, 'mean_') and scaler.mean_ is not None and \
                             hasattr(scaler, 'with_mean') and scaler.with_mean: # and not scaler.with_std (or scale_ is None, centering only)
                            data[col_name] = data[col_name] + scaler.mean_[idx_in_scaler]
                        # If neither with_std nor with_mean, or scale_/mean_ are None, no transformation was effectively done by the scaler for this feature.
                        
                    except ValueError:
                        # This case should ideally not be reached if features_to_transform is built correctly
                        st.warning(f"Colonne {col_name} prévue pour la dénormalisation mais non trouvée dans les caractéristiques du scaler.")
                    except IndexError:
                        st.error(f"Erreur d'index pour la colonne {col_name} lors de la dénormalisation. "
                                 f"Index: {idx_in_scaler}, Taille de scale/mean: {len(scaler.scale_ if hasattr(scaler, 'scale_') and scaler.scale_ is not None else scaler.mean_)}")
        return data
    
    st.session_state['unscaled_data'] = reverse_pretreatment(st.session_state.get('data').drop(columns=['SK_ID_CURR']))

    st.subheader("Analyse univariée du positionnement du client par rapport à la population",
                 help="Permet de comparer la valeur client sur une caractéristique donnée par rapport au reste de la clientèle.")
    st.selectbox(
        label="Sélectionnez une caractéristique",
        options=st.session_state.get('top_shap_features')['Feature'].tolist(),
        index=st.session_state.get('top_shap_features')['Feature'].tolist().index(st.session_state.get('feature_selection')) if st.session_state.get('feature_selection') in st.session_state.get('top_shap_features')['Feature'].tolist() else 0,
        key='feature_selection',
        help="Sélectionnez une caractéristique pour voir son impact sur la prédiction du client sélectionné. \
            Les caractéristiques sont triées par valeur SHAP décroissante pour toute la population. \
                Les valeurs SHAP indiquent l'importance de chaque caractéristique client dans la prédiction du modèle."
    )

    feature = st.session_state.get('feature_selection')
    if 'graph_mode' not in st.session_state:
        st.session_state.graph_mode = 'histogramme'

    st.radio(
        label="Mode d'affichage",
        options=['histogramme', 'boxplot'],
        format_func=str.capitalize,
        index=0,
        key='graph_mode',
        help="Mode d'affichage des valeurs SHAP. \
        Le mode 'Histogramme' affiche un histogramme de la distribution de la caractéristique, \
        tandis que le mode 'Boxplot' affiche un boxplot de la distribution de la caractéristique."
    )

    # Déterminer si la caractéristique sélectionnée a un impact positif ou négatif sur la prédiction du client
    if feature in st.session_state.get('top_shap_features')['Feature'].tolist():
        feature_shap_value = st.session_state.get('top_shap_features').loc[st.session_state.get('top_shap_features')['Feature'] == feature, 'SHAP Value'].values[0]
        if feature_shap_value > 0:
            st.success(f"La caractéristique {feature} a un impact positif sur la prédiction du client n°{st.session_state.id_client}.")
        else:
            st.error(f"La caractéristique {feature} a un impact négatif sur la prédiction du client n°{st.session_state.id_client}.")

    if not feature:
        st.warning("Veuillez sélectionner une caractéristique.")
        return

    # Afficher la distribution de la caractéristique sélectionnée en affichant le positionnement du client avec une ligne rouge
    st.subheader(f"Positionnement du client n°{st.session_state.id_client} sur la caractéristique {feature}",
                 help="Distribution de la caractéristique sélectionnée pour toute la population. \
                    Le client sélectionné est représenté par une ligne rouge.")
    # On récupère la valeur de la caractéristique pour le client sélectionné
    client_value = st.session_state.get('unscaled_data').loc[st.session_state.get('data')['SK_ID_CURR'] == st.session_state.id_client, feature].values[0]
    # On récupère la distribution de la caractéristique pour toute la population
    population_values = st.session_state.get('unscaled_data')[feature].values
    # On crée un DataFrame avec la distribution de la caractéristique
    population_df = pd.DataFrame({
        'Caractéristique': population_values
    })
    def get_distribution_chart(mode):
        """
        Fonction pour créer le graphique de distribution de la caractéristique sélectionnée.
        """
        if mode == 'histogramme':
            # On crée un histogramme de la distribution de la caractéristique avec Altair
            chart = alt.Chart(population_df).mark_bar().encode(
                x=alt.X('Caractéristique', bin=alt.Bin(maxbins=20), title=feature), # Histogramme avec 20 bins
                y=alt.Y('count()', title='Nombre de clients'), # Nombre de clients
                tooltip=['count()'] # Info-bulle avec le nombre de clients
            ).properties(
            )
            # On ajoute une ligne rouge pour le client sélectionné
            chart += alt.Chart(pd.DataFrame({'Caractéristique': [client_value]})).mark_rule(color='red').encode(
                x='Caractéristique:Q', # Valeur de la caractéristique pour le client sélectionné
                tooltip=['Caractéristique'] # Info-bulle avec la valeur de la caractéristique
            ).properties()
        elif mode == 'boxplot':
            # On crée un boxplot de la distribution de la caractéristique avec Altair
            chart = alt.Chart(population_df).mark_boxplot(extent='min-max').encode(
                y=alt.Y('Caractéristique:Q', title=feature), # Boxplot
                tooltip=[alt.Tooltip('Caractéristique:Q', title=feature)]
            ).properties(
                title=f"Distribution de la caractéristique {feature} pour toute la population", # Titre du graphique
                width=300 
            )
            # On ajoute une ligne rouge pour le client sélectionné
            client_line = alt.Chart(pd.DataFrame({'Caractéristique': [client_value]})).mark_rule(color='red', size=2).encode(
                y='Caractéristique:Q',
                tooltip=[alt.Tooltip('Caractéristique:Q', title=f"Valeur client ({st.session_state.id_client})")]
            )
            chart = chart + client_line
        return chart
    # On appelle la fonction pour créer le graphique de distribution de la feature sélectionnée
    with st.spinner("Calcul de la distribution de la caractéristique...", show_time=True):
        st.altair_chart(get_distribution_chart(st.session_state.get('graph_mode')), use_container_width=True)
    mean_value = st.session_state.get('unscaled_data')[feature].mean()
    std_value = st.session_state.get('unscaled_data')[feature].std()
    client_value = st.session_state.get('unscaled_data').loc[st.session_state.get('data')['SK_ID_CURR'] == st.session_state.id_client, feature].values[0]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label=f"Valeur du client n°{st.session_state.id_client}",
            value=f"{client_value:.2f}",
            help="Valeur de la caractéristique pour le client sélectionné. \
                Cette valeur est calculée en prenant la valeur de la caractéristique pour le client sélectionné.\
                La flèche indique si la valeur du client est supérieure ou inférieure à la médiane."
        )
    with col2:
        st.metric(
            label="Moyenne de la caractéristique",
            value=f"{mean_value:.2f}",
            delta = f"{client_value - mean_value:.2f}",
            delta_color="normal",
            help="Moyenne de la caractéristique pour toute la population. \
                **La moyenne d'une série statistique est la somme des valeurs divisée par le nombre de valeurs. \
                On peut interpréter la moyenne comme la valeur qu'auraient toutes les valeurs si elles étaient égales.**\
                La flèche indique si la valeur du client est supérieure ou inférieure à la moyenne."
        )
    with col3:
        std_variation = (client_value - mean_value) / std_value
        st.metric(
            label="Écart-type de la caractéristique",
            value=f"{std_value:.2f}",
            delta = f"{std_variation:.2f}",
            delta_color="normal",
            help="Écart-type de la caractéristique pour toute la population. \
                **L'écart-type d'une série statistique est une mesure de la dispersion des valeurs autour de la moyenne. \
                Il indique, en moyenne, à quelle distance les valeurs sont éloignées de la moyenne.**\
                La flèche indique de combien de variations d'écart-type la valeur du client est éloignée de la moyenne."
        )
    
    feature_description = st.session_state.get('columns_descriptions')
    print(feature)
    print(feature_description['Features'].unique())
    if feature in feature_description['Features'].unique():
        # On récupère la description de la caractéristique sélectionnée
        feature_description = feature_description.loc[feature_description['Features'] == feature, 'Description'].values[0]
    else:
        feature_description = "No description is available for this characteristic."    

    # Traduction de la description en français via Google Translate
    async def translate_text(text: str, source_language: str = 'en', target_language: str = 'fr'):
        """
        Fonction pour traduire le texte en français.
        """
        async with Translator() as translator:
            translated_text = await translator.translate(text, src=source_language, dest=target_language)
        # On retourne le texte traduit
        translated_text = translated_text.text
        return translated_text
    
    st.subheader(f"Description de la caractéristique {feature}",
                 help="Description de la caractéristique sélectionnée. \
                    Cette description est extraite du fichier HomeCredit_columns_description.csv. \
                    La description est traduite en français via Google Translate.")
    translated_text = asyncio.run(translate_text(feature_description, source_language='en', target_language='fr'))
    st.write(translated_text)

def home():
    st.title("Accueil")
    st.subheader("Bienvenue sur le tableau de bord de prédiction")
    st.write("Ce tableau de bord vous permet de faire des prédictions à l'aide du modèle chargé depuis S3.")
    # Afficher le contenu du README.md
    with open('./README.md', 'r', encoding='utf-8') as f:
        readme_content = f.read()
    st.write(readme_content)

st.set_page_config(
    page_title="Tableau de bord de prédiction",
    page_icon=":guardsman:",
    layout="wide",
    initial_sidebar_state="expanded"
)

pg = st.navigation([
    st.Page(home, title= "Accueil", icon="🏠"),
    st.Page(prediction, title= "Prédiction", icon="📈"),
    st.Page(positionnement, title= "Positionnement", icon="👤")
])

init_session_state_variables()

# Infos client sidebar
with st.sidebar:
    st.title("Informations client")
    st.write("Sélectionnez un client pour voir ses informations.")
    if st.session_state.get('init_done'):
        # Id du client
        st.session_state.id_client = st.selectbox(
            label="Sélectionnez l'ID du client",
            options=st.session_state.get('all_ids'),
            index=0
        )
        st.session_state.index_client = st.session_state.get('all_ids').index(st.session_state.id_client)

        client_data = st.session_state.get('raw_data').loc[st.session_state.get('raw_data')['SK_ID_CURR'] == st.session_state.id_client, :]
        gender = "Femme" if client_data['CODE_GENDER'].values[0] == 'F' else "Homme"
        how_many_children = client_data['CNT_CHILDREN'].values[0]
        adressing_client = {
            "generic": "un homme" if gender == "Homme" else "une femme",
            "pronom": "il" if gender == "Homme" else "elle",
            "children": "a un enfant" if how_many_children == 1 else (f"a {how_many_children} enfants" if how_many_children > 1 else "n'a pas d'enfants")
            }
        age = client_data['DAYS_BIRTH'].values[0] // -365
        income = client_data['AMT_INCOME_TOTAL'].values[0]
        payment_rate = client_data['AMT_CREDIT'].values[0] / client_data['AMT_ANNUITY'].values[0]
        employment_duration = client_data['DAYS_EMPLOYED'].values[0] // -30
        
        if employment_duration < 0:
            employment_duration = 0
        else:
            employment_duration = int(employment_duration)
        how_many_docs_provided = 0
        total_docs = 0
        for col in client_data.columns:
            if 'FLAG_DOCUMENT' in col:
                how_many_docs_provided += client_data[col].values[0]
                total_docs += 1
        how_many_docs_provided = int(how_many_docs_provided)
        total_docs = int(total_docs)

        st.subheader(f"Informations sur le client n°{st.session_state.id_client}:")
        st.write(f"Le client n°{st.session_state.id_client} est {adressing_client['generic']} de {age} ans. {adressing_client['pronom'].capitalize()} {adressing_client['children']}.")
        st.write(f"{adressing_client['pronom'].capitalize()} a un revenu de {income:.2f} $ et un taux d'endettement de {payment_rate:.2f}%.")
        st.write(f"{adressing_client['pronom'].capitalize()} a {employment_duration} mois d'ancienneté dans son emploi actuel.")
        st.write(f"{adressing_client['pronom'].capitalize()} a fourni {how_many_docs_provided} document(s) sur les {total_docs} demandés.")

if __name__ == "__main__":
    pg.run()