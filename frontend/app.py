from flask import Flask, render_template, request, redirect, flash, send_from_directory, session
import mysql.connector
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, render_template, request, redirect, session
from werkzeug.security import generate_password_hash, check_password_hash
import mysql.connector

app = Flask(__name__)
app.secret_key = 'admin'

# Database connection
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    port="3306",
    database='medical_plant'
)
mycursor = mydb.cursor()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transformations for main model (no normalization)
image_transform_main = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

# Image transformations for normalization (irrelevant/relevant classification)
image_transform_normalization = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the MobileNetModel class for relevance/irrelevance classification
class MobileNetModel(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetModel, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        num_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier[1] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.mobilenet(x)

# Load the relevant/irrelevant model
irrelevant_model_path = 'mobilenet_irrelevent.pt'
irrelevant_model = MobileNetModel(num_classes=2)
irrelevant_model.load_state_dict(torch.load(irrelevant_model_path))
irrelevant_model = irrelevant_model.to(device)
irrelevant_model.eval()
# Define the MobileNetRNN class for 40 class classification
class RNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        out = self.fc(rnn_out[:, -1, :])
        return F.log_softmax(out, dim=1)

class MobileNetRNN(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetRNN, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        self.mobilenet.classifier = nn.Identity()
        
        self.feature_dim = 1280
        self.rnn_input_dim = 1280
        self.rnn_hidden_dim = 512
        self.rnn_output_dim = num_classes
        
        self.rnn = RNNClassifier(self.rnn_input_dim, self.rnn_hidden_dim, self.rnn_output_dim)
    
    def forward(self, x):
        features = self.mobilenet(x)
        features = features.view(features.size(0), -1)
        features = features.unsqueeze(1)
        
        output = self.rnn(features)
        return output
# Load the model
num_classes = 40  # Adjust according to your number of classes
model = MobileNetRNN(num_classes=num_classes)
model.load_state_dict(torch.load('best_model_rnn.pth'))
model = model.to(device)
model.eval()

# Load class names
dataset_root = 'Medicinal plant dataset'
class_names = sorted([d.name for d in os.scandir(dataset_root) if d.is_dir()])



# Helper functions
def predict_relevance(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image_transform_normalization(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = irrelevant_model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

def predict_classification(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image_transform_main(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted_class = torch.max(outputs, 1)
    plant_name = class_names[predicted_class.item()]

    # Initialize description and uses
    description = "No description available"
    uses = "No uses available"

    # Provide details for each plant class
    if plant_name == 'Wood_sorel':
        description = "Wood sorrel is a wild herb with a sour taste, often used in salads."
        uses = "Used in salads, soups, and as a garnish."
    elif plant_name == 'Brahmi':
        description = "Brahmi is an herb commonly used in Ayurvedic medicine for enhancing memory and reducing stress."
        uses = "Used as a memory enhancer, stress reliever, and general tonic."
    elif plant_name == 'Basale':
        description = "Basale, or basil, is an aromatic herb used in cooking and for medicinal purposes."
        uses = "Used in cooking, especially in Italian dishes, and for its medicinal properties such as aiding digestion."
    elif plant_name == 'Lemon_grass':
        description = "Lemongrass is a tropical plant used for its citrus flavor and medicinal properties."
        uses = "Used in teas, soups, and as a flavoring in various dishes."
    elif plant_name == 'Lemon':
        description = "Lemon is a citrus fruit known for its tangy flavor and high vitamin C content."
        uses = "Used in cooking, baking, and as a flavoring in drinks and foods."
    elif plant_name == 'Insulin':
        description = "Insulin is a hormone used in diabetes management."
        uses = "Used in medical treatments for diabetes to regulate blood sugar levels."
    elif plant_name == 'Amruta_Balli':
        description = "Amruta Balli is a plant used in traditional medicine, known for its health benefits."
        uses = "Used in traditional remedies to improve health and treat various ailments."
    elif plant_name == 'Betel':
        description = "Betel is a plant whose leaves are used in chewing and for medicinal purposes."
        uses = "Used in chewing for its stimulant effects and in traditional medicine."
    elif plant_name == 'Castor':
        description = "Castor is a plant known for its oil, which has various industrial and medicinal uses."
        uses = "Used to produce castor oil for industrial applications and as a laxative."
    elif plant_name == 'Ashoka':
        description = "Ashoka is a tree valued in traditional medicine for its therapeutic properties."
        uses = "Used in traditional medicine to treat gynecological disorders and other conditions."
    elif plant_name == 'Aloevera':
        description = "Aloe Vera is a succulent plant known for its soothing and healing properties."
        uses = "Used in skin care products and as a remedy for burns and wounds."
    elif plant_name == 'Tulasi':
        description = "Tulasi, or Holy Basil, is a revered plant in Ayurveda with various health benefits."
        uses = "Used in traditional medicine to boost immunity and as a general tonic."
    elif plant_name == 'Henna':
        description = "Henna is a plant known for its dyeing properties, used in body art and hair coloring."
        uses = "Used for creating temporary body art and coloring hair."
    elif plant_name == 'Curry_Leaf':
        description = "Curry leaf is an aromatic herb used in Indian cuisine and traditional medicine."
        uses = "Used in cooking for its distinctive flavor and in medicine for digestive health."
    elif plant_name == 'Arali':
        description = "Arali, or Nerium Oleander, is a plant known for its ornamental value and medicinal properties."
        uses = "Used in traditional medicine and as an ornamental plant."
    elif plant_name == 'Hibiscus':
        description = "Hibiscus is a plant with vibrant flowers used in teas and traditional medicine."
        uses = "Used in herbal teas and for its antioxidant properties."
    elif plant_name == 'Betel_Nut':
        description = "Betel nut is the seed of the Areca palm, used in chewing and for medicinal purposes."
        uses = "Used in traditional chewing practices and in some medicinal applications."
    elif plant_name == 'Neem':
        description = "Neem is a tree known for its antimicrobial and medicinal properties."
        uses = "Used in traditional medicine and personal care products for its health benefits."
    elif plant_name == 'Jasmine':
        description = "Jasmine is a fragrant flower used in perfumes and traditional medicine."
        uses = "Used in perfumes, teas, and for its calming effects in traditional medicine."
    elif plant_name == 'Nithyapushpa':
        description = "Nithyapushpa, also known as the everlasting flower, is used for its aesthetic and medicinal properties."
        uses = "Used in traditional medicine and for its ornamental value."
    elif plant_name == 'Mint':
        description = "Mint is a fragrant herb used in cooking, teas, and for medicinal purposes."
        uses = "Used in cooking, teas, and to soothe digestive issues."
    elif plant_name == 'Nooni':
        description = "Nooni is a plant known for its traditional medicinal uses."
        uses = "Used in traditional medicine to treat various health conditions."
    elif plant_name == 'Pomegranate':
        description = "Pomegranate is a fruit known for its health benefits and rich flavor."
        uses = "Consumed fresh or used in juices, and known for its antioxidant properties."
    elif plant_name == 'Pepper':
        description = "Pepper is a spice used globally to add flavor to dishes and for medicinal purposes."
        uses = "Used as a spice in cooking and in traditional medicine for digestive health."
    elif plant_name == 'Geranium':
        description = "Geranium is a plant known for its aromatic leaves and ornamental value."
        uses = "Used in aromatherapy and as an ornamental plant."
    elif plant_name == 'Mango':
        description = "Mango is a tropical fruit known for its sweet flavor and nutritional benefits."
        uses = "Consumed fresh, in juices, or used in cooking."
    elif plant_name == 'Honge':
        description = "Honge, or Pongamia, is a plant used in traditional medicine and as a biofuel source."
        uses = "Used in traditional medicine and for producing biofuels."
    elif plant_name == 'Amla':
        description = "Amla is a fruit known for its high vitamin C content and health benefits."
        uses = "Used in traditional medicine to boost immunity and in health supplements."
    elif plant_name == 'Ekka':
        description = "Ekka is a plant used in traditional medicine with various health benefits."
        uses = "Used in traditional remedies to improve health and treat ailments."
    elif plant_name == 'Raktachandini':
        description = "Raktachandini is a plant used in traditional medicine for its therapeutic properties."
        uses = "Used in traditional medicine for treating skin conditions and other ailments."
    elif plant_name == 'Rose':
        description = "Rose is a flower known for its beauty and fragrance, used in various products."
        uses = "Used in perfumes, cosmetics, and traditional medicine for its soothing properties."
    elif plant_name == 'Ashwagandha':
        description = "Ashwagandha is an adaptogenic herb used in Ayurvedic medicine to reduce stress."
        uses = "Used to reduce stress, improve energy levels, and support overall health."
    elif plant_name == 'Gauva':
        description = "Gauva, or guava, is a tropical fruit known for its sweet flavor and health benefits."
        uses = "Consumed fresh, in juices, or used in cooking for its nutritional benefits."
    elif plant_name == 'Ganike':
        description = "Ganike is a plant with traditional uses in herbal medicine."
        uses = "Used in traditional medicine to treat various health conditions."
    elif plant_name == 'Avocado':
        description = "Avocado is a fruit known for its creamy texture and nutritional benefits."
        uses = "Consumed fresh, in salads, or used in cooking for its healthy fats."
    elif plant_name == 'Sapota':
        description = "Sapota, or chikoo, is a tropical fruit known for its sweet flavor and nutritional value."
        uses = "Consumed fresh or used in desserts and smoothies."
    elif plant_name == 'Doddapatre':
        description = "Doddapatre, or Panax, is a plant used in traditional medicine for its health benefits."
        uses = "Used in traditional remedies to improve health and treat ailments."
    elif plant_name == 'Nagadali':
        description = "Nagadali is a plant with traditional uses in herbal medicine."
        uses = "Used in traditional medicine to treat various health conditions."
    elif plant_name == 'Pappaya':
        description = "Papaya is a tropical fruit known for its digestive enzymes and health benefits."
        uses = "Consumed fresh or used in smoothies and salads for its digestive benefits."
    elif plant_name == 'Bamboo':
        description = "Bamboo is a fast-growing plant used in construction, furniture, and as a food source in some cultures."
        uses = "Used in construction, furniture making, and as a food source in some cultures."

    return plant_name, description, uses


def map_prediction_to_label(prediction):
    label_mapping = {0: "relevant", 1: "irrelevant"}
    return label_mapping.get(prediction, "Unknown")

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/about2')
def about2():
    return render_template('about2.html')


@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form.get('email')
        password = request.form.get('password')
        c_password = request.form.get('confirm_password')
        address = request.form.get('address')
        mobile_number = request.form.get('mobile_number')

        # # Validate form input
        # if not email or not password or not c_password or not address or not mobile_number:
        #     return render_template('register.html', message="All fields are required!")

        if password != c_password:
            return render_template('register.html', message="Confirm password does not match!")

        if len(mobile_number) != 10 or not mobile_number.isdigit():
            return render_template('register.html', message="Mobile number must be exactly 10 digits!")

        # Check if email exists
        query = "SELECT email FROM users WHERE email = %s"
        mycursor.execute(query, (email,))
        email_data = mycursor.fetchone()

        if email_data:
            return render_template('register.html', message="This email ID already exists!")

        # Hash password before storing
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        # Insert user into the database
        query = "INSERT INTO users (email, password, address, mobile_number) VALUES (%s, %s, %s, %s)"
        try:
            mycursor.execute(query, (email, hashed_password, address, mobile_number))
            mydb.commit()
        except mysql.connector.Error as err:
            return render_template('register.html', message="Database error: " + str(err))

        return render_template('login.html', message="Successfully Registered!")
    
    return render_template('register.html')

@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        query = "SELECT email, password FROM users WHERE email = %s"
        mycursor.execute(query, (email,))
        user = mycursor.fetchone()
        if user and check_password_hash(user[1], password):
            session['user_email'] = email
            return redirect('/home')
        return render_template('login.html', message="Invalid email or password!")
    return render_template('login.html')

@app.route('/home')
def home():
    if 'user_email' not in session:
        return redirect('/login')
    return render_template('home.html')

@app.route("/upload", methods=["POST", "GET"])
def upload():
    if request.method == 'POST':
        myfile = request.files['file']
        fn = myfile.filename
        if myfile and myfile.filename.split('.')[-1].lower() in ['jpg', 'png', 'jpeg', 'jfif']:
            mypath = os.path.join('static/uploaded_images', myfile.filename)
            myfile.save(mypath)
            
            # First, check if the image is relevant
            relevance = predict_relevance(mypath)
            if relevance == 0:  # If relevant
                classification, description, uses = predict_classification(mypath)
                return render_template("upload.html", mypath=mypath, prediction=classification, description=description,uses=uses, fn=fn)
            else:  # If not relevant
                return render_template("upload.html", mypath=mypath, prediction="This image is not relevant to this project!", fn=fn)
        
        flash("Only image formats are accepted", "danger")
    return render_template('upload.html')

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("static/uploaded_images", filename)

if __name__ == '__main__':
    app.run(debug=True)
