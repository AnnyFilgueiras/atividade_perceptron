from data import data_train, data_test
from perceptron import train, predict

def main ():
    print("--- Iniciando treinamento ---")
    weightTemperature, weightHumidity, bias, epochs = train (data_train)
    print ("--- Treinamento concluído ---")
    print (f"Épocas executadas: {epochs}")
    print (f"Peso final da temperatura: {weightTemperature}")
    print (f"Peso final da umidade: {weightHumidity}")
    print (f"Viés final: {bias}")

    print("--- Conjunto de testes ---")
    predictCorrects = 0
    for d in data_test:
        predicted = predict(d["temperature"], d["humidity"], weightTemperature, weightHumidity, bias)
        if predicted == d["class"]:
            predictCorrects += 1
            
    accuracy = (predictCorrects / len(data_test)) * 100
    print(f"Acurácia no conjunto de teste: {accuracy}%\n")

    print("--- Classificação de novos valores ---")
    
    while True:

        temperature = float(input("Digite a temperatura (°C): "))
                
        humidity = float(input("Digite a umidade relativa (%): "))
            
        result = predict(temperature, humidity, weightTemperature, weightHumidity, bias)

        if (result == 1):
            label = "ALERTA (1)"
        else:
            label = "NORMAL (0)"
            
        print(f"Classificação prevista: {label}\n")

if __name__ == "__main__":
    main()