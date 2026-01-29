# 🚀 ComfyUI Any-Device Offload

**Force any Model, VAE, or CLIP to any GPU or CPU—and keep it there (or don't).**

This custom node gives you total control over where your models run and where they live in memory. It solves common "Out of Memory" (OOM) errors, enables multi-GPU workflows, and fixes persistent crashes when trying to run modern workflows (like Z_Image/Flux) on CPUs or secondary GPUs.

<img width="309" height="220" alt="image" src="https://github.com/user-attachments/assets/c02ec349-d704-43d5-aaa3-c08bbdc658ab" />

<img width="2184" height="906" alt="image" src="https://github.com/user-attachments/assets/912b0f0a-4dc5-4ff6-8101-9281bf935100" />

<img width="1517" height="496" alt="image" src="https://github.com/user-attachments/assets/3b98fafd-f6db-4f94-b2e3-c452eac42dd4" />

<img width="309" height="220" alt="image" src="https://github.com/user-attachments/assets/607eabf1-385c-4ad7-84b8-a2c9b3a8d6da" />


* Z_Image_Turbo - 2 steps only using CPU test(Looks like CPU can run with less steps)
<img width="516" height="370" alt="image" src="https://github.com/user-attachments/assets/937d5c2a-25f6-4c5f-b968-f4cf98628cfc" />

### ✨ Key Features

* **👇 Manual Device Selection:** Force specific models to run on `cuda:0`, `cuda:1`, `cpu`, or Mac `mps`.
* **🧠 Intelligent VRAM Management:**
    * **Keep in Memory (True):** Pin models to VRAM for instant switching.
    * **Keep in Memory (False):** The "Kill Switch." Automatically unloads the Model, CLIP, and VAE from VRAM immediately after generation finishes. It aggressively triggers garbage collection (`gc.collect`) and clears the CUDA cache (`empty_cache`) to free up space for other apps or workflows.
* **🛠️ VAE Patcher (The Crash Fixer):**
    * Fixes `RuntimeError: Input type (c10::BFloat16) and bias type (float) should be the same`.
    * Fixes `CuDNN error: GET was unable to find an engine to execute this computation`.
    * Automatically converts inputs and weights to matching types (Float32/BFloat16) on the fly without breaking the workflow.
* **🛡️ CPU "Safe Mode":**
    * Automatically intercepts **xFormers** calls on CPU (which usually crash with "No operator found") and redirects them to standard PyTorch attention.
    * Auto-casts Float16 inputs to Float32 when running on CPU to prevent "mat1 and mat2" dtype errors.

---

### 📦 Installation

1.  Navigate to your ComfyUI custom nodes directory:
    ```bash
    cd ComfyUI/custom_nodes/
    ```
2.  Clone this repository:
    ```bash
    git clone https://github.com/FearL0rd/ComfyUI-AnyDeviceOffload.git
    ```
3.  **Restart ComfyUI.**

---

### 📖 Usage

Add the node **"Offload Anything (GPU/CPU)"** to your workflow (found under `utils/hardware`).

#### **Inputs**
1.  **target_device:**
    * Select the hardware you want to use (e.g., `cuda:0`, `cuda:1`, `cpu`).
    * *Note: If you select CPU, the node automatically enables "Safe Mode" to prevent xFormers crashes.*
2.  **vae_mode:**
    * **`Original` (Default):** Uses the standard VAE behavior.
    * **`Vae Patched`:** **⚠️ IMPORTANT:** Select this if you get **black images**, **NaN errors**, or **PyTorch crashes** (especially with Flux, SD3, or secondary GPUs). It forces Float32 precision and auto-casts incoming tensors to ensure compatibility.
3.  **keep_in_memory:**
    * **`True`:** Models stay loaded on the device (faster for repeated runs).
    * **`False`:** Models are aggressively purged from VRAM immediately after the image is saved. Great for low-VRAM cards.

#### **Connections**
* Connect your **MODEL**, **CLIP**, and/or **VAE** into the inputs.
* Connect the outputs to your KSampler or VAE Decode.

---

### 💡 Troubleshooting / FAQ

**Q: My generation crashes with "Input type and bias type should be the same".**
> **A:** Switch **`vae_mode`** to **`Vae Patched`**. This forces the VAE to handle the data type mismatch (usually caused by Flux/SD3 generating BFloat16 latents while the VAE expects Float32).

**Q: I get an xFormers error when using CPU.**
> **A:** This node automatically patches xFormers to work on CPU. If you still see it, ensure you are running the latest version of this node and have restarted ComfyUI.

**Q: Why is my GPU memory not clearing?**
> **A:** Uncheck **`keep_in_memory`**. This activates the "Kill Switch" which forces a `torch.cuda.empty_cache()` immediately after the VAE finishes decoding.

---
### 📦 Release Notes
 ## v1.0.0 (First Release)
   This custom node gives you total control over where your models run and where they live in memory. It solves common (OOM) errors, enables multi-GPU workflows, and fixes persistent crashes when trying to run modern workflows (like Z_Image/Flux) on CPUs or secondary GPUs.
 ## v1.0.1 (Making the default selections)
   making the default selections set to help beginners in ComfyUI
 ## v1.0.2 (Add legacy GPU support (SM < 8.0) and fix dynamic device switching)
   This update introduces a robust "Runtime Device Guard" and intelligent hardware detection to solve crashes on older GPUs and multi-GPU workflows

---

### 👨‍💻 Credits

Created to solve complex memory management and precision mismatches in ComfyUI multi-device workflows.

*License: MIT*
