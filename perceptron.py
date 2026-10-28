def normalize (temperature, humidity):
    temperatureNormalized = temperature / 40
    humidityNormalized    = humidity / 100
    return temperatureNormalized, humidityNormalized

def predict (temperature, humidity, weightTemperature, weightHumidity, bias):
    (temperatureNormalized, humidityNormalized) = normalize (temperature, humidity)
    weightedSum = weightTemperature * temperatureNormalized + weightHumidity * humidityNormalized + bias

    if (weightedSum >= 0):
        return 1
    else:
        return 0

def train (data_train, rate = 0.1, maxEpoch = 1000):
    weightTemperature = 0.0
    weightHumidity    = 0.0
    bias              = 0.0
    
    for currentEpoch in range (maxEpoch):
        errorsInEpoch = 0

        for d in data_train:
            outputPredict = predict (d["temperature"], d["humidity"], weightTemperature, weightHumidity, bias)
            error         = d["class"] - outputPredict

            if (error != 0):
                (temperatureNormalized, humidityNormalized) = normalize (d["temperature"], d["humidity"])
                weightTemperature += rate * error * temperatureNormalized
                weightHumidity    += rate * error * humidityNormalized
                bias              += rate * error
                errorsInEpoch += 1

        if (errorsInEpoch == 0):
            break

    return weightTemperature, weightHumidity, bias, currentEpoch + 1