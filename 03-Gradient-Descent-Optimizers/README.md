# 🚀 Advanced Linear Regression with Adaptive Optimizers From Scratch

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.20%2B-darkgreen.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4%2B-orange.svg?style=for-the-badge&logo=python&logoColor=white)](https://matplotlib.org/)
[![Framework](https://img.shields.io/badge/From--Scratch-No--Frameworks-red.svg?style=for-the-badge)](https://github.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

تطبيق برمجى متكامل واحترافي لنموذج **الانحدار الخطي (Linear Regression)** تم بناؤه من الصفر تماماً باستخدام مكتبة **NumPy** فقط دون الاعتماد على أي مكتبات تعلم آلة جاهزة (مثل Scikit-Learn). 

يركز هذا المستودع على تقديم مقارنة تطبيقية ورياضية عميقة بين أشهر خوارزميات المحسنات الذكية (**Momentum**, **RMSprop**, **Adam**) عند دمجها مع آلية **Mini-batch Gradient Descent** لتسريع التخطي والوصول للحل الأمثل (Global Minimum).

---

## 📌 جدول المحتويات / Table of Contents
- [📖 نبذة عن المشروع / Overview](#-نبذة-عن-المشروع--overview)
- [📐 الشرح الرياضي والمعادلات / Mathematical Framework](#-الشرح-الرياضي-والمعادلات--mathematical-framework)
  - [1. دالة التكلفة والغراديان / Cost & Gradients](#1-دالة-التكلفة-والغراديان--cost--gradients)
  - [2. محسن الزخم / Momentum Optimizer](#2-محسن-الزخم--momentum-optimizer)
  - [3. محسن RMSprop Optimizer](#3-محسن-rmsprop-optimizer)
  - [4. محسن Adam Optimizer](#4-محسن-adam-optimizer)
- [⚡ مقارنة سريعة بين المحسنات / Optimizers Comparison](#-مقارنة-سريعة-بين-المحسنات--optimizers-comparison)
- [✨ المميزات البرمجية / Key Features](#-المميزات-البرمجية--key-features)
- [🚀 دليل التشغيل والتنصيب / Installation & Quick Start](#-دليل-التشغيل-والتنصيب--installation--quick-start)
- [📊 المخرجات والرسوم البيانية / Interactive Dashboard & Outputs](#-المخرجات-والرسوم-البيانية--interactive-dashboard--outputs)
- [⚙️ الأوزان الفائقة الافتراضية / Hyperparameter Configurations](#%EF%B8%8F-الأوزان-الفائقة-الافتراضية--hyperparameter-configurations)

---

## 📖 نبذة عن المشروع / Overview
في المشاريع الحقيقية والشبكات العصبية العميقة، يعاني الانحدار التدريجي التقليدي (Gradient Descent) من البطء الشديد والتذبذب عند التعامل مع البيانات الكبيرة أو الأسطح الرياضية المعقدة. 

يوفر هذا المشروع فئة برمجية (Class) مرنة ومكتوبة بأسلوب **Object-Oriented Programming (OOP)** تتيح للمطورين والباحثين فهم كيفية عمل خوارزميات ضبط معدل التعلم المتكيف (Adaptive Learning Rate) برمجياً ورؤية تأثيرها المباشر على سرعة التقارب (Convergence).

---

## 📐 الشرح الرياضي والمعادلات / Mathematical Framework

### 1. دالة التكلفة والغراديان / Cost & Gradients
الهدف هو تقليل دالة الخطأ **متوسط مربعات الخطأ الموزون (MSE Cost Function)**:

$$J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2$$

حيث يتم حساب التوقع لكل نقطة عبر المعادلة الخطية:  
$$\hat{y}^{(i)} = w \cdot x^{(i)} + b$$

يتم حساب المشتقات الجزئية (Gradients) لكل دفعة بيانات (Mini-batch) لتوجيه التحديث:
$$\frac{\partial J}{\partial w} = dw = \frac{1}{m_{batch}} \sum_{i=1}^{m_{batch}} (\hat{y}^{(i)} - y^{(i)}) \cdot x^{(i)}$$
$$\frac{\partial J}{\partial b} = db = \frac{1}{m_{batch}} \sum_{i=1}^{m_{batch}} (\hat{y}^{(i)} - y^{(i)})$$

---

### 2. محسن الزخم / Momentum Optimizer
يعتمد على محاكاة حركة الكرة المتدحرجة، حيث يجمع العزم من التحديثات السابقة لتسريع الحركة في الاتجاه الصحيح وتقليل الاهتزازات العشوائية الحادة.

* **تحديث السرعة المتجهة (Velocity Accumulation):**
  $$v_{dw} = \beta_1 \cdot v_{dw} + (1 - \beta_1) \cdot dw$$
  $$v_{db} = \beta_1 \cdot v_{db} + (1 - \beta_1) \cdot db$$

* **تحديث الأوزان والإنحياز:**
  $$w = w - \eta \cdot v_{dw}$$
  $$b = b - \eta \cdot v_{db}$$

---

### 3. محسن RMSprop Optimizer
يقوم بتعديل معدل التعلم تلقائياً لكل متغير على حدة عبر قسمة الغراديان على الجذر التربيعي للمتوسط المتحرك للمربعات السابقة. هذا يقلل من الخطوات في الاتجاهات ذات التذبذب العالي.

* **حساب التباين المتحرك (Moving Average of Squared Gradients):**
  $$v_{dw} = \beta_2 \cdot v_{dw} + (1 - \beta_2) \cdot dw^2$$
  $$v_{db} = \beta_2 \cdot v_{db} + (1 - \beta_2) \cdot db^2$$

* **تحديث الأوزان والإنحياز:**
  $$w = w - \frac{\eta}{\sqrt{v_{dw}} + \epsilon} \cdot dw$$
  $$b = b - \frac{\eta}{\sqrt{v_{db}} + \epsilon} \cdot db$$

---

### 4. محسن Adam Optimizer
يعد المحسن القياسي الأكثر كفاءة في العالم اليوم؛ فهو يدمج بين مزايا العزم الأول (Momentum) والعزم الثاني (RMSprop) مع تطبيق خطوة **تصحيح الانحياز (Bias Correction)** للتحكم في دقة البدايات الناتجة عن تصفير المصفوفات.

* **العزم الأول والثاني (First & Second Moments):**
  $$v_{dw1} = \beta_1 \cdot v_{dw1} + (1 - \beta_1) \cdot dw$$
  $$v_{db1} = \beta_1 \cdot v_{db1} + (1 - \beta_1) \cdot db$$
  $$v_{dw2} = \beta_2 \cdot v_{dw2} + (1 - \beta_2) \cdot dw^2$$
  $$v_{db2} = \beta_2 \cdot v_{db2} + (1 - \beta_2) \cdot db^2$$

* **تصحيح الانحياز المعتمد على الخطوة الزمنية $t$:**
  $$\hat{v}_{dw1} = \frac{v_{dw1}}{1 - \beta_1^t}, \quad \hat{v}_{db1} = \frac{v_{db1}}{1 - \beta_1^t}$$
  $$\hat{v}_{dw2} = \frac{v_{dw2}}{1 - \beta_2^t}, \quad \hat{v}_{db2} = \frac{v_{db2}}{1 - \beta_2^t}$$

* **التحديث النهائي للمتغيرات:**
  $$w = w - \frac{\eta}{\sqrt{\hat{v}_{dw2}} + \epsilon} \cdot \hat{v}_{dw1}$$
  $$b = b - \frac{\eta}{\sqrt{\hat{v}_{db2}} + \epsilon} \cdot \hat{v}_{db1}$$

---

## ⚡ مقارنة سريعة بين المحسنات / Optimizers Comparison

| المحسن (Optimizer) | الميزة الأساسية | التكيف مع المعاملات | معالجة التذبذب |
| :--- | :--- | :--- | :--- |
| **Momentum** | تراكم السرعة بالاتجاه الصحيح | ثابت لجميع المتغيرات | ممتاز |
| **RMSprop** | يمنع انفجار/تلاشي معدل التعلم | متكيف (Adaptive) | جيد جداً |
| **Adam** | يجمع بين العزم والتباين + تصحيح الانحياز | متكيف وذكي جداً | الأفضل كفاءة |

---

## ✨ المميزات البرمجية / Key Features
- **Pure OOP Architecture:** كود نظيف، منظم، وقابل للتوسيع بالكامل لفصل العمليات الرياضية.
- **Dynamic Mini-batch Scaling:** إمكانية التحكم الكامل بحجم دفعة البيانات (Batch Size) من خلال الواجهة.
- **Smart Early Stopping:** حلقة التدريب ذكية وتتوقف تلقائياً بمجرد وصول قيمة دالة الخطأ لنسبة مستهدفة أقل من `1e-5` لتوفير موارد المعالجة وتجنب الـ Overfitting.
- **Auto-generated Dashboard:** يقوم الكود تلقائياً بحفظ لوحة بيانات متكاملة بصيغة `output.png` بدقة عالية وثيم داكن عصري.

---

## 🚀 دليل التشغيل والتنصيب / Installation & Quick Start

### 1. استنساخ المستودع وتثبيت المكتبات
```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME

# Install dependencies
pip install numpy matplotlib
