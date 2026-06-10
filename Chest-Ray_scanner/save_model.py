"""
save_model.py
=============
شغّل الـ script ده في نهاية الـ notebook عشان تحفظ الـ model
"""

import os

os.makedirs("model", exist_ok=True)

# بعد ما تخلص التدريب في الـ notebook، أضف السطر ده:
model_pretrained.save("model/chest_xray_model.h5")

print("✅ Model saved to model/chest_xray_model.h5")
