# Classification-of-Medicinal-Plant-Leafs-Using-Ensembling-Techniques-Deep-Learning-Computer-Vision
The project introduces a deep learning system that integrates MobileNetV2 with Variational Mode Decomposition (VMD) for identifying medicinal plants from leaf images.
MobileNetV2: A lightweight, high-performance convolutional neural network is utilized for its ability to deliver fast, resource-efficient processing, making it suitable for real-world applications on mobile and embedded devices.
Variational Mode Decomposition (VMD): VMD is applied to decompose leaf images into intrinsic mode functions. This process enhances the extraction of discriminative features, boosting the classification performance of the model.
Dataset Preparation: A comprehensive dataset of medicinal plant leaf images is collected and preprocessed. Data augmentation techniques are used to improve the robustness and diversity of training data.
Workflow:
Leaf images are preprocessed and subjected to VMD for feature enhancement.
These enhanced images are then fed into MobileNetV2 for classification.
Transfer learning leverages pre-trained weights, leading to faster convergence and enhanced accuracy.
Evaluation: The model’s performance is evaluated using metrics such as accuracy, precision, and recall, ensuring reliability and strong generalization across different plant species and conditions.
Impact:
Automating the identification process addresses challenges posed by traditional manual methods, reducing human error and subjectivity.
The system supports researchers, herbalists, and practitioners by providing quick and accurate plant identification.
Significance: The integration of VMD with MobileNetV2 establishes a new benchmark for both efficiency and accuracy in medicinal plant identification. The approach is easily scalable and adaptable to other classification tasks in botanical research and agriculture.
