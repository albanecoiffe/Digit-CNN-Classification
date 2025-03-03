import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
from streamlit_drawable_canvas import st_canvas
import matplotlib.cm as cm


# Charger le modèle entraîné
@st.cache_resource
def load_trained_model():
    return load_model("mnist_cnn_model.h5")  # Assurez-vous que le fichier est dans le dossier du script

model = load_trained_model()

# Fonction pour prédire une image
def predict_digit(img_array):
    prediction = model.predict(img_array)
    return np.argmax(prediction, axis=1)[0], np.max(prediction)

# Fonction pour afficher une image et sa prédiction
def display_prediction(img_array, predicted_class, confidence):
    fig, ax = plt.subplots()
    ax.imshow(img_array.reshape(28, 28), cmap="gray")
    ax.axis("off")
    ax.set_title(f"Predicted: {predicted_class} (Confidence: {confidence:.2f})")
    st.pyplot(fig)

def get_model_summary(model):
    """Capture le résumé du modèle et le retourne sous forme de texte"""
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + "\n"))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string

def display_top_predictions(probas):
    """Affiche les 10 prédictions les plus probables sous forme de graphique"""
    top_indices = np.argsort(probas[0])[::-1][:10]  # Récupérer les 10 meilleures classes
    top_probs = probas[0][top_indices]  # Probabilités associées

    # Définir une palette de couleurs (ex: dégradé de bleu à rouge)
    colors = cm.viridis(np.linspace(0, 1, 10))  # Utilisation du colormap 'viridis'

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(range(10), top_probs[::-1], align="center", color=colors)
    ax.set_yticks(range(10))
    ax.set_yticklabels(top_indices[::-1])
    ax.set_xlabel("Confidence Score")
    ax.set_title("Top 10 Predictions")

    # Ajouter les scores de confiance au-dessus des barres
    for bar, prob in zip(bars, top_probs[::-1]):
        ax.text(bar.get_width() + 0.02, bar.get_y() + 0.3, f"{prob:.2f}", fontsize=10, color="white", weight="bold")

    st.pyplot(fig)

def draw_digit():
    st.subheader("Draw a digit (0-9) below")

    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=10,
        stroke_color="white",
        background_color="black",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas"
    )

    if st.button("Predict Drawn Digit"):
        if canvas_result.image_data is not None:
            img = Image.fromarray(canvas_result.image_data.astype("uint8"))
            img = img.convert("L").resize((28, 28))
            img_array = np.array(img) / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)

            pred_class, confidence = predict_digit(img_array)
            display_prediction(img_array, pred_class, confidence)


# Interface Streamlit
st.title("MNIST Digit Classification")
st.write("Upload an image of a handwritten digit (28x28 pixels) for classification.")

# Sidebar pour la navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Choose an option:", ["Test MNIST Image", "Upload an Image", "Draw a Digit", "Model Info"])

# Tester une image du dataset MNIST
if option == "Test MNIST Image":
    st.subheader("Test a Random Image from MNIST")
    st.write(
        "Click the button below to generate a **random handwritten digit** from the MNIST dataset. "
        "The model will classify the image and display its confidence scores for each possible digit."
    )

    if st.button("Generate Random Image"):
        # Charger le dataset MNIST
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        index = np.random.randint(0, len(x_test))
        img = x_test[index]

        # Normalisation et reshape pour le modèle
        img_array = img.reshape(1, 28, 28, 1) / 255.0  

        # Prédiction du modèle
        predicted_class, confidence = predict_digit(img_array)

        # Afficher l'image avec la prédiction
        st.write("### 🔍 Image Selected:")
        display_prediction(img, predicted_class, confidence)

        # Afficher le vrai label
        st.write(f"✔️ **True label**: {y_test[index]}")

        # Explication des résultats
        st.write(
            "📊 Below, you can see the **Top 10 predictions** of the model, sorted by confidence score. "
            "The highest bar represents the model's best guess."
        )

        # Afficher les 10 meilleures prédictions sous forme de graphique
        y_pred = model.predict(img_array)
        display_top_predictions(y_pred)

        # Ajouter une note explicative
        st.info(
            "ℹ️ The confidence score represents the probability assigned by the model to each digit. "
            "Higher values indicate a stronger prediction."
        )


# Uploader une image personnalisée
elif option == "Upload an Image":
    st.subheader("Upload an Image for Classification")
    uploaded_file = st.file_uploader("Upload an image (28x28 grayscale)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Charger et prétraiter l'image
        img = Image.open(uploaded_file).convert("L")  # Convertir en niveau de gris
        img = img.resize((28, 28))  # Redimensionner
        img_array = np.array(img) / 255.0  # Normalisation
        img_array = img_array.reshape(1, 28, 28, 1)  # Reshape pour le modèle

        # Faire la prédiction
        predicted_class, confidence = predict_digit(img_array)

        # Afficher l'image et la prédiction
        display_prediction(img_array, predicted_class, confidence)

if option == "Model Info":
    st.subheader("Model Architecture")
    st.write("How does this CNN work?")
    st.write("""
    This model is a **Convolutional Neural Network (CNN)** trained on the MNIST dataset to recognize handwritten digits (0-9). 
    It uses:
    - **Convolutional Layers** to detect patterns like curves and edges.
    - **Pooling Layers** to reduce the image size while keeping key features.
    - **Fully Connected Layers** to make the final classification.
    
    The model achieves high accuracy on digit classification!
    """)
    model_summary = get_model_summary(model)  # Obtenir le résumé sous forme de texte
    st.text(model_summary)  # Afficher correctement dans Streamlit

if option == "Draw a Digit":
    draw_digit()
