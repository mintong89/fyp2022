import {
  Button,
  ChakraProvider,
  Container,
  Divider,
  HStack,
  Radio,
  RadioGroup,
  Text,
  Textarea,
  VStack,
} from '@chakra-ui/react'
import { useState } from 'react'

const App = () => {
  const senValue = ['negative', 'neutral', 'positive']

  const [modelSelect, setModelSelect] = useState('bert')

  const [isLoading, setLoading] = useState(false)
  const [loadingText, setLoadingText] = useState(<></>)

  const [senSentence, senSenSentence] = useState('')
  const [senPolarity, setSenPolarity] = useState('')

  const accuracy = {
    bert: 0.7894,
    xlmr: 0.8138,
    gpt2: 0.5812,
  }

  const callPython = ({
    inputs = {},
    outputs = (res) => {},
    inputsLoading = <></>,
    outputsLoading = <></>,
  }) => {
    let data = new FormData()
    data.append('data', JSON.stringify(inputs))
    setLoading(true)
    setLoadingText(inputsLoading)

    fetch('http://localhost:8000', {
      method: 'POST',
      body: data,
    })
      .then((res) => res.json())
      .then((res) => {
        outputs(res)
        setLoadingText(outputsLoading)
      })
  }

  return (
    <ChakraProvider>
      <Container maxW="6xl" my={4}>
        <VStack align="flex-start" spacing={4}>
          <Text fontWeight="bold" fontSize="2xl">
            Sentiment Classification
          </Text>
          <Textarea
            value={senSentence}
            onChange={(e) => senSenSentence(e.target.value)}
            placeholder="Enter sentences here..."
            resize="none"
          />
          <RadioGroup
            onChange={(value) => {
              setModelSelect(value)
              setSenPolarity('')
              callPython({
                inputs: {
                  function: 'change_model',
                  model_type: value,
                },
                outputs: () => {
                  setLoading(false)
                },
                inputsLoading: <>Loading model...</>,
                outputsLoading: <>Loading completed.</>,
              })
            }}
            value={modelSelect}
          >
            <HStack spacing={4}>
              <Radio value="bert">BERT+LE</Radio>
              <Radio value="xlmr">XLM-R+LE</Radio>
              <Radio value="gpt2">GPT2+LE</Radio>
            </HStack>
          </RadioGroup>
          <HStack spacing={4}>
            <Button
              colorScheme="teal"
              onClick={() => {
                setSenPolarity('')
                callPython({
                  inputs: {
                    function: 'run_script',
                    sentence: senSentence,
                  },
                  outputs: (res) => {
                    setSenPolarity(res?.output ?? '')
                    setLoading(false)
                  },
                  inputsLoading: <>Running script...</>,
                  outputsLoading: <>Running completed.</>,
                })
              }}
              isLoading={isLoading}
              isDisabled={senSentence.length === 0}
            >
              Run
            </Button>
            <Text>{loadingText}</Text>
          </HStack>
          <Text>
            {senPolarity && (
              <>
                <b>Result:</b> The sentence is a {senValue[senPolarity]}{' '}
                sentence.
              </>
            )}
          </Text>
        </VStack>

        <Divider my={2} />

        <Text>
          Expected Accuracy: {(accuracy[modelSelect] * 100).toFixed(2)}%
        </Text>
      </Container>
    </ChakraProvider>
  )
}

export default App
