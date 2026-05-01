### **TECHNICAL DOCUMENTATION / IMPLEMENTATION CHAPTER**

**3.0 SYSTEM IMPLEMENTATION**

**3.1 System Architecture**

The proposed lightweight knowledge-gap mapping system follows a modular pipeline architecture consisting of five main components:

1. **Data Ingestion Layer** – Handles raw interaction logs from different platforms with flexible column mapping.
2. **Preprocessing and Feature Engineering Module** – Transforms raw logs into meaningful behavioral features.
3. **Knowledge Gap Inference Engine** – Uses a lightweight, interpretable Decision Tree model.
4. **Explainability and Evaluation Layer** – Generates human-readable explanations and remediation suggestions.
5. **User Interface** – Interactive Streamlit dashboard for educators and students.

The entire system is designed to run locally on modest hardware (standard laptop with 8GB RAM) without requiring cloud infrastructure or paid tools.

**3.2 Technology Stack**

- **Programming Language**: Python 3.10
- **Data Processing**: pandas, NumPy
- **Machine Learning**: scikit-learn (DecisionTreeClassifier)
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Web Interface**: Streamlit
- **Development Environment**: Anaconda

All tools are open-source, aligning with the goal of zero-cost implementation suitable for resource-constrained educational institutions.

**3.3 Data Ingestion and Standardization**

The system accepts student interaction logs in CSV format. To ensure robustness across different learning platforms, a flexible ingestion layer was implemented (`LogIngestion` class). 

Key features:
- Automatic column mapping for common OULAD fields (`id_student`, `date`, `sum_click`, `activity_type`)
- Manual column mapping interface in the dashboard for custom logs
- Standardization into a unified schema: `student_id`, `timestamp`, `activity_type`, `concept`, `num_interactions`

**3.4 Feature Engineering**

Behavioral and temporal features were engineered from interaction logs, including:

- Total interactions per concept
- Interaction consistency and span
- Average daily engagement
- Interaction ratio relative to student activity

These features capture patterns indicative of learning difficulty and conceptual misunderstanding.

**3.5 Model Development and Explainability (Core Contribution)**

A **Decision Tree Classifier** (max_depth=5) was chosen as the core model for the following reasons:

- **High Interpretability**: Unlike black-box models (e.g., neural networks), decision trees allow full transparency of decision paths.
- **Explainability**: Feature importance scores and tree visualization clearly show which behavioral patterns contribute most to knowledge gap predictions.
- **Lightweight**: Low computational cost, making it suitable for deployment in under-resourced environments.
- **Good Performance**: Achieved **87.46% accuracy** on the test set using OULAD data.

The model predicts the probability of a knowledge gap for each student-concept pair. Explainability is further enhanced through:
- Feature importance ranking
- Rule-based severity assessment (Low, Medium, High)
- Human-readable evidence generation (e.g., “Very low interactions + Low consistency across days”)

**3.6 Remediation and Knowledge Gap Mapping**

For each detected gap, the system generates:
- Severity level
- Supporting behavioral evidence
- Personalized remediation suggestions based on severity

This transforms raw predictions into pedagogically useful outputs for both students and instructors.

**3.7 User Interface**

An interactive Streamlit dashboard provides:
- One-click demo execution
- Upload interface with column mapping
- Visual overview of risk levels (bar charts)
- Detailed per-student gap analysis
- Export functionality for student reports

**3.8 Evaluation**

The system was evaluated using:
- Predictive performance (Accuracy = 87.46%)
- Interpretability of explanations
- Practical usability through the dashboard interface




### **4.0 RESULTS AND DISCUSSION**

**4.1 Model Performance**

The lightweight Decision Tree model achieved an accuracy of **87.46%** on the test set using the Open University Learning Analytics Dataset (OULAD), specifically the AAA-2013J module presentation. This performance demonstrates that meaningful patterns indicative of knowledge gaps can be effectively learned from student interaction logs alone.

The confusion matrix and classification report showed balanced performance between the “No Gap” and “Knowledge Gap” classes, indicating that the model does not suffer from severe class imbalance bias despite the natural distribution in the dataset.

**4.2 Explainability Analysis**

Explainability was the central focus of this implementation. The Decision Tree model provides several layers of transparency:

1. **Global Explainability**: Feature importance ranking clearly identifies which behavioral features contribute most to knowledge gap predictions. The most influential features typically include:
   - Interaction consistency
   - Total interactions per concept
   - Average daily engagement
   - Interaction ratio relative to overall student activity

2. **Local Explainability**: For each student-concept pair, the system generates human-readable evidence such as “Very low interactions + Low consistency across days” or “High model confidence (0.82)”.

3. **Severity Classification**: Gaps are categorized into Low, Medium, and High severity using a combination of model probability and behavioral thresholds. This allows instructors to prioritize interventions effectively.

This level of explainability addresses one of the major limitations of many existing learning analytics systems — the “black box” problem — and significantly increases trust and pedagogical usefulness.

**4.3 Knowledge Gap Mapping Output**

The system successfully transforms raw interaction logs into structured knowledge gap maps. Each map contains:
- Specific concepts/activities where gaps exist
- Gap probability and severity level
- Supporting behavioral evidence
- Personalized remediation suggestions (e.g., “Urgent intervention needed – Schedule one-on-one session with tutor”)

These outputs are presented through an intuitive Streamlit dashboard, making them accessible to both instructors and students.

**4.4 Practical Implications**

The prototype demonstrates that advanced learning analytics capabilities can be achieved without heavy computational resources or proprietary platforms. This is particularly relevant for educational institutions in developing contexts like Nigeria, where access to expensive cloud-based AI tools may be limited.

The emphasis on open-source tools and local deployment aligns with the goal of democratizing personalized learning support.

**4.5 Limitations**

- The current prototype primarily uses `activity_type` as a proxy for fine-grained concepts due to limitations in the OULAD dataset structure.
- Full custom column mapping for arbitrary log formats is partially implemented (UI exists, but deep integration is still being refined).
- The system currently focuses on post-hoc analysis rather than real-time gap detection.
- Evaluation was conducted on a single course module (AAA-2013J). Broader validation across multiple courses would strengthen generalizability.



### **5.0 CONCLUSION AND FUTURE WORK**

**5.1 Conclusion**

This project successfully designed and implemented a **lightweight knowledge-gap mapping system** for personalized learning using student interaction logs. The system addresses a critical challenge in digital education: the inability of many existing learning analytics tools to provide timely, interpretable, and actionable insights into learners’ conceptual weaknesses.

By leveraging open-source tools and a transparent Decision Tree model, the prototype achieved a respectable accuracy of **87.46%** while maintaining high explainability. The system goes beyond simple performance prediction by generating structured knowledge gap maps that include severity assessment and personalized remediation suggestions. This bridges the gap between raw interaction data and pedagogically meaningful intervention.

The emphasis on simplicity, low computational requirements, and zero-cost implementation makes the system particularly suitable for resource-constrained educational environments, including institutions in Nigeria and other developing contexts.

Key achievements include:
- Development of a modular and extensible pipeline
- Implementation of explainable AI techniques suitable for educational use
- Creation of an interactive Streamlit dashboard for practical usability
- Successful integration of data ingestion, feature engineering, modeling, and visualization components

This work contributes to the growing field of learning analytics by demonstrating that effective knowledge-gap detection does not necessarily require complex deep learning models or expensive infrastructure.

**5.2 Future Work**

Several enhancements can further strengthen the system:

1. **Improved Concept Mapping**: Move beyond activity_type proxy to finer-grained concept detection by integrating course content metadata or topic modeling techniques.

2. **Real-time Processing**: Extend the system to support streaming logs for near real-time gap detection and intervention.

3. **Advanced Upload Support**: Fully implement dynamic column mapping and automatic format detection for logs from popular platforms such as Moodle, Google Classroom, and custom LMS.

4. **Multi-course Validation**: Evaluate the model across all OULAD modules and, where possible, on local institutional datasets to improve generalizability.

5. **User Studies**: Conduct usability testing with actual instructors and students to assess the pedagogical usefulness of the generated explanations and remediation suggestions.

6. **Fairness and Bias Analysis**: Incorporate fairness metrics to ensure the system does not disproportionately disadvantage particular student groups.

7. **Mobile-Friendly Interface**: Optimize the dashboard for mobile devices to increase accessibility for educators and students.

**5.3 Recommendations**

For successful adoption in Nigerian higher education institutions, future iterations should prioritize:
- Integration with existing university LMS platforms
- Support for low-bandwidth environments
- Training materials for instructors on interpreting and acting upon the system’s outputs


