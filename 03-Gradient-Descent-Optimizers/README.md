
# 🚀 الانحدار الخطي المتقدم مع المحسنات الذكية | Advanced Linear Regression with Optimizers From Scratch

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.20%2B-darkgreen.svg?style=for-the-badge&logo=numpy)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4%2B-orange.svg?style=for-the-badge)](https://matplotlib.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

مستودع برمجيات احترافي يقدم تطبيقاً متكاملاً لنموذج **الانحدار الخطي (Linear Regression)** من الصفر تماماً بدون استخدام مكتبات تعلم آلة جاهزة. يركز المشروع على مقارنة وشرح سلوك أشهر خوارزميات التحسين الذكية (**Momentum**, **RMSprop**, **Adam**) المطبقة فوق آلية **Mini-batch Gradient Descent**.

An industrial-grade, from-scratch implementation of a **Linear Regression** model using only NumPy. This repository serves as a comprehensive reference for understanding and benchmarking advanced optimization techniques (**Momentum**, **RMSprop**, and **Adam**) integrated with **Mini-batch Gradient Descent**.

---

## 📌 جدول المحتويات / Table of Contents
- [📖 مقدمة عن المشروع / Overview](#-مقدمة-عن-المشروع--overview)
- [📐 الشرح الرياضي والمعادلات / Mathematical Breakdown](#-الشرح-الرياضي-والمعادلات--mathematical-breakdown)
  - [1. دالة التكلفة والغراديان / Cost & Gradients](#1-دالة-التكلفة-والغراديان--cost--gradients)
  - [2. محسن الزخم / Momentum Optimizer](#2-محسن-الزخم--momentum-optimizer)
  - [3. محسن RMSprop Optimizer](#3-محسن-rmsprop-optimizer)
  - [4. محسن Adam Optimizer](#4-محسن-adam-optimizer)
- [✨ المميزات البرمجية / Core Features](#-المميزات-البرمجية--core-features)
- [🛠️ هيكلية الكود / Code Structure](#%EF%B8%8F-هيكلية-الكود--code-structure)
- [🚀 دليل التشغيل والتنصيب / Installation & Usage](#-دليل-التشغيل-والتنصيب--installation--usage)
- [📊 المخرجات والرسوم البيانية / Outputs & Visualizations](#-المخرجات-والرسوم-البيانية--outputs--visualizations)
- [⚙️ الإعدادات الافتراضية / Hyperparameters](#%EF%B8%8F-الإعدادات-الافتراضية--hyperparameters)

---

## 📖 مقدمة عن المشروع / Overview
في تطبيقات الذكاء الاصطناعي، لا يكفي استخدام خوارزمية الانحدار التدريجي التقليدية (GD) لأنها تعاني من البطء الشديد والتذبذب الحاد عند التعامل مع السطوح غير المنتظمة أو البيانات الضخمة. 

هذا المشروع يوفر فئة (Class) برمجية مرنة بلغة `Python` تتيح للمستخدم إدخال بياناته وتجربة ثلاثة من أقوى المحسنات الرياضية التي تعتمد عليها الشبكات العصبية العميقة اليوم، مما يسهل فهم هندسة الخوارزميات وتأثير الأوزان الفائقة (Hyperparameters).

---

## 📐 الشرح الرياضي والمعادلات / Mathematical Breakdown

### 1. دالة التكلفة والغراديان / Cost & Gradients
الهدف من الانحدار الخطي هو إيجاد أفضل قيم للوزن ($w$) والانحياز ($b$) لتقليل دالة التكلفة **متوسط مربعات الخطأ (MSE)**:

$$J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)^2$$

حيث أن القيمة المتنبأ بها هي:  
$$\hat{y}^{(i)} = w \cdot x^{(i)} + b$$

يتم حساب المشتقات الجزئية (Gradients) لكل دفعة بيانات (Mini-batch) كالتالي:
$$\frac{\partial J}{\partial w} = dw = \frac{1}{m_{batch}} \sum_{i=1}^{m_{batch}} (\hat{y}^{(i)} - y^{(i)}) \cdot x^{(i)}$$
$$\frac{\partial J}{\partial b} = db = \frac{1}{m_{batch}} \sum_{i=1}^{m_{batch}} (\hat{y}^{(i)} - y^{(i)})$$

---

### 2. محسن الزخم / Momentum Optimizer
يعمل هذا المحسن مثل كرة تتدحرج من أعلى التل، حيث يراكم السرعة المتجهة (Velocity) في الاتجاهات المستقرة ويقلل التذبذب العشوائي في الاتجاهات الأخرى.

* **معادلة تحديث السرعة المتجهة (Velocity Update):**
  $$v_{dw} = \beta_1 \cdot v_{dw} + (1 - \beta_1) \cdot dw$$
  $$v_{db} = \beta_1 \cdot v_{db} + (1 - \beta_1) \cdot db$$

* **معادلة تحديث المعاملات (Parameter Update):**
  $$w = w - \eta \cdot v_{dw}$$
  $$b = b - \eta \cdot v_{db}$$

*حيث $\beta_1$ هو معامل الزخم (عادة ما يكون 0.9)، و $\eta$ هو معدل التعلم (Learning Rate).*

---

### 3. محسن RMSprop Optimizer
(Root Mean Square Propagation) يقوم بحل مشكلة تلاشي أو انفجار معدل التعلم عن طريق مواءمة خطوة التعلم بشكل منفصل لكل معلمة (Adaptive Learning Rate). يتم تقليل الخطوة للمتغيرات ذات الاهتزازات الكبيرة وزيادتها للمتغيرات المستقرة.

* **معادلة المتوسط المتحرك لمربع الغراديان (Exponential Moving Average):**
  $$v_{dw} = \beta_2 \cdot v_{dw} + (1 - \beta_2) \cdot dw^2$$
  $$v_{db} = \beta_2 \cdot v_{db} + (1 - \beta_2) \cdot db^2$$

* **معادلة تحديث المعاملات:**
  $$w = w - \frac{\eta}{\sqrt{v_{dw}} + \epsilon} \cdot dw$$
  $$b = b - \frac{\eta}{\sqrt{v_{db}} + \epsilon} \cdot db$$

*حيث $\epsilon$ هو رقم صغير جداً (مثلاً $10^{-8}$) لمنع القسمة على صفر.*

---

### 4. محسن Adam Optimizer
(Adaptive Moment Estimation) يعتبر المحسن الأقوى والأكثر استخداماً عالمياً، حيث يدمج بين فكرتي **Momentum** (العزم الأول - المتوسط) و **RMSprop** (العزم الثاني - التباين)، مع إضافة خطوة لتصحيح الانحياز (Bias Correction) للتغلب على مشكلة بدء العزوم من الصفر.

* **حساب العزم الأول والثاني (First and Second Moments):**
  $$v_1 = \beta_1 \cdot v_1 + (1 - \beta_1) \cdot g$$
  $$v_2 = \beta_2 \cdot v_2 + (1 - \beta_2) \cdot g^2$$

* **تصحيح الانحياز (Bias Correction):**
  $$\hat{v}_{1} = \frac{v_1}{1 - \beta_1^t}, \quad \hat{v}_{2} = \frac{v_2}{1 - \beta_2^t}$$

* **معادلة تحديث المعاملات النهائية:**
  $$w = w - \frac{\eta}{\sqrt{\hat{v}_{dw2}} + \epsilon} \cdot \hat{v}_{dw1}$$
  $$b = b - \frac{\eta}{\sqrt{\hat{v}_{db2}} + \epsilon} \cdot \hat{v}_{db1}$$

*حيث $t$ يمثل خطوة التحديث الحالية (Iteration Counter).*

---

## ✨ المميزات البرمجية / Core Features
- **تطبيق نقي بالكامل (Pure OOP):** كود منظم داخل فئة برمجية واحدة قابلة لإعادة الاستخدام.
- **إيقاف مبكر ذكي (Smart Early Stopping):** يتوقف النموذج تلقائياً بمجرد الوصول لنسبة خطأ مقبولة مستهدفاً سرعة المعالجة وعدم إضاعة دورات تدريبية غير ضرورية.
- **تفاعلية كاملة (Fully Interactive):** يستقبل البيانات وأحجام الـ batches واختيار المحسن مباشرة من الـ Terminal.
- **توليد لوحات تحكم بيانية (Visual Dashboard Output):** يقوم تلقائياً بإنشاء رسم بياني احترافي وعالي الجودة وحفظه بصيغة `output.png`.

---

## 🛠️ هيكلية الكود / Code Structure

تتكون الفئة `linear_regression_model` من التوابع الأساسية التالية:
* `__init__`: لتهيئة البارامترات الفائقة الأساسية ($\eta, \text{epochs}, \beta_1, \beta_2, \epsilon$).
* `fit_momentum`: حلقة تدريب تعتمد على خوارزمية الزخم.
* `fit_rmsprop`: حلقة تدريب تعتمد على خوارزمية RMSprop.
* `fit_adam`: حلقة تدريب تعتمد على خوارزمية Adam الشاملة وتصحيح الانحياز.
* `predict`: لحساب التوقعات بناءً على المعاملات النهائية وطباعتها بشكل منسق.
* `plot`: لتوليد الرسومات البيانية ثنائية المحاور بدقة عالية وثيم مظلم.
* `run_model`: واجهة التحكم الرئيسية التي تنسق تشغيل المهام بالترتيب.

---

## 🚀 دليل التشغيل والتنصيب / Installation & Usage

1. قم بعمل استنساخ (Clone) للمستودع:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
   cd YOUR_REPO_NAME
قم بتثبيت المكتبات المطلوبة:

Bash
pip install numpy matplotlib
قم بتشغيل الملف البرمجي:

Bash
python main.py
مثال على المدخلات التفاعلية (Interactive CLI Example):
Plaintext
Enter the number of data points from n to t separated by comma: 1,2,3,4,5
Enter the corresponding y values: 3,5,7,9,11
Enter the new x values for prediction: 6,10,15
Enter the number of data points for each batch: 2
Enter the name of optimizer (momentum, rmsprop, adam): adam
📊 المخرجات والرسوم البيانية / Outputs & Visualizations
بعد انتهاء تشغيل النموذج بنجاح، ستظهر لك النتائج في الـ Terminal كالتالي:

Plaintext
==================================================
the prediction of model is 
equation of line is y = 2.0000 * x + 1.0000
Predictions:
x = 6  -->  y = 13.00
x = 10  -->  y = 21.00
x = 15  -->  y = 31.00
==================================================
final value
Final w:  2.0000
Final b:  1.0000
Final cost:  0.0000
Training stopped at epoch: 42
كما سيقوم البرنامج بحفظ لوحة بيانية باسم output.png في نفس المسار:

المخطط الأول (Linear Regression): يعرض نقاط البيانات الحقيقية باللون الأصفر والخط المستقيم الفائز باللون الأحمر.

المخطط الثاني (Cost Function): يوضح الهبوط الأسّي لدالة التكلفة مع توضيح نقطة الإيقاف المبكر بدقة.

⚙️ الإعدادات الافتراضية / Hyperparameters
القيم التي تم ضبط الكائن بها افتراضياً في نهاية الكود تعد المعيار الصناعي الأمثل (Golden Standard) لمعظم المهام:

Learning Rate (eta) = 0.01

Max Epochs = 1000

Beta 1 (Momentum) = 0.9

Beta 2 (RMSprop) = 0.999

Epsilon = 1e-8

🤝 المساهمة (Contributing): ترحب المساهمات دائماً! لا تتردد في فتح Issue أو إرسال Pull Request لتطوير هذا النموذج.
