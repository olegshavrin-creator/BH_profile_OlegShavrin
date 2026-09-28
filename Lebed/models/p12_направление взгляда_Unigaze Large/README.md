# UniGaze Large — local inference

Локальная версия inference-скрипта из Colab для запуска в VS Code.

Модель: `unigaze_l16_joint` (UniGaze Large / L16).

## Что изменено относительно ноутбука

Алгоритмическая логика не менялась:

- загрузка `unigaze_l16_joint`;
- `face_alignment`;
- оценка положения головы;
- нормализация изображения для UniGaze;
- расчёт gaze vector / pitch-yaw;
- классификация пяти направлений;
- HOLD / RELEASE / REARM логика событий;
- русский overlay-счётчик;
- запись итогового MP4 и CSV.

Изменено только окружение:

- пути `/content/...` заменены на локальные пути относительно репозитория;
- поиск шрифта работает на Windows, Linux/WSL и macOS;
- папка `output` создаётся автоматически;
- официальный репозиторий UniGaze ожидается в `third_party/UniGaze`.

## Структура

```text
unigaze_large_local/
├─ unigaze_large_inference.py
├─ requirements.txt
├─ requirements-windows-cuda118.txt
├─ setup_windows.ps1
├─ README.md
├─ .gitignore
├─ input/
│  └─ input.mp4          # добавить самостоятельно
├─ output/               # создаваемые MP4/CSV
└─ third_party/
   └─ UniGaze/           # setup_windows.ps1 клонирует автоматически
```

## Windows + NVIDIA + VS Code

### 1. Требования

- Windows 10/11
- Python 3.10
- Git
- актуальный NVIDIA driver
- VS Code + Python extension

### 2. Автоматическая установка

Открой PowerShell в корне репозитория:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_windows.ps1
```

Скрипт:

1. создаст `.venv`;
2. установит CUDA 11.8 build PyTorch 2.0.1 / torchvision 0.15.2;
3. установит `timm==0.3.2` и остальные Python-зависимости;
4. клонирует официальный UniGaze в `third_party/UniGaze`;
5. создаст `input` и `output`.

Такой PyTorch/timm набор соответствует easy-install инструкции UniGaze.

### 3. VS Code

Открой папку проекта в VS Code.

Выбери интерпретатор:

```text
.venv\Scripts\python.exe
```

### 4. Входное видео

Помести видео сюда:

```text
input\input.mp4
```

### 5. Запуск

```powershell
python unigaze_large_inference.py
```

Результаты:

```text
output\input_large_unigaze.mp4
output\input_large_unigaze.csv
```

## Проверка GPU

Скрипт сохраняет исходную логику выбора устройства:

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

Проверка:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Ожидается:

```text
True
<название NVIDIA GPU>
```

Если CUDA недоступна, inference всё равно сохранит исходное поведение и переключится на CPU, но будет существенно медленнее.

## Первый запуск

`unigaze.load(...)` скачивает pretrained-веса UniGaze при первом использовании. Поэтому для первого запуска нужен интернет. После загрузки веса берутся из локального cache.

`face_alignment` также может скачать pretrained-модели при первом запуске.

## Другие платформы / другой PyTorch

`requirements-windows-cuda118.txt` сделан именно для воспроизводимого Windows/NVIDIA окружения.

Если используется другой CUDA/PyTorch:

1. создай `.venv`;
2. установи подходящий PyTorch вручную;
3. установи:

```powershell
pip install timm==0.3.2
pip install -r requirements.txt
```

## Что не коммитить в GitHub

`.gitignore` уже исключает:

- `.venv`;
- входные интервью;
- обработанные видео;
- локальную копию `third_party/UniGaze`;
- служебные файлы VS Code/Python.

UniGaze подключается как сторонняя зависимость, а не копируется в этот репозиторий.

## Лицензия UniGaze

Официальный UniGaze использует MG-NC-RAI-2.0 (`NonCommercial`).
Если продукт планируется использовать коммерчески, условия коммерческого использования модели нужно согласовать с правообладателем отдельно.

Официальный проект:
https://github.com/ut-vision/UniGaze

PyPI:
https://pypi.org/project/unigaze/
