import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import telebot
from telebot import types

# Инициализация бота с вашим токеном
bot = telebot.TeleBot("7516439034:AAF59uroLK2LSSrcTjHeT5oWZ05stCazCJw")

# Название моделей
financial_model_name = 'test_trainer/alignment-checkpoint-18'
travel_model_name = 'travel_trainer/checkpoint-15'

# Переменная состояния для отслеживания выбранного ассистента
user_state = {}

# Устройство (CPU)
device = torch.device("cpu")

# Ленивая загрузка токенизаторов и моделей
def get_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    return tokenizer, model

financial_tokenizer, financial_model = get_model_and_tokenizer(financial_model_name)
travel_tokenizer, travel_model = get_model_and_tokenizer(travel_model_name)

# Конфигурация генерации
generation_config = GenerationConfig(
    max_new_tokens=50,  # Уменьшение максимального количества токенов
    do_sample=False,    # Отключение случайной генерации
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    pad_token_id=financial_tokenizer.eos_token_id
)

# Обработчик стартовой команды и меню выбора ассистента
@bot.message_handler(commands=['start'])
def start(message):
    markup = types.ReplyKeyboardMarkup(row_width=2)
    btn1 = types.KeyboardButton('Финансовый')
    btn2 = types.KeyboardButton('Тревел')
    markup.add(btn1, btn2)
    bot.send_message(message.chat.id, "Выберите ассистента:", reply_markup=markup)

# Обработчик выбора ассистента
@bot.message_handler(func=lambda message: message.text in ['Финансовый', 'Тревел'])
def choose_assistant(message):
    if message.text == 'Финансовый':
        user_state[message.chat.id] = 'financial'
        bot.send_message(message.chat.id, "Привет! Я ваш финансовый ассистент. Чем могу помочь?")
    elif message.text == 'Тревел':
        user_state[message.chat.id] = 'travel'
        bot.send_message(message.chat.id, "Привет! Я ваш тревел ассистент. Чем могу помочь?")

# Обработчик сообщений после выбора ассистента
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    if message.chat.id in user_state:
        try:
            if user_state[message.chat.id] == 'financial':
                tokenizer, model = financial_tokenizer, financial_model
            elif user_state[message.chat.id] == 'travel':
                tokenizer, model = travel_tokenizer, travel_model
            else:
                bot.send_message(message.chat.id, "Пожалуйста, выберите ассистента, используя команду /start.")
                return

            user_input = message.text  # Получение сообщения пользователя
            inputs = tokenizer(user_input, return_tensors="pt").to(device)

            # Генерация ответа
            outputs = model.generate(**inputs, generation_config=generation_config)
            if outputs is not None:
                response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                bot.reply_to(message, response)
            else:
                bot.reply_to(message, "Модель не вернула ответ.")
        
        except Exception as e:
            bot.reply_to(message, f"Произошла ошибка при обработке запроса: {e}")
    else:
        bot.send_message(message.chat.id, "Пожалуйста, выберите ассистента, используя команду /start.")

# Запуск бота
bot.polling()
