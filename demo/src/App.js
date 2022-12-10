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
  const [senClass, setSenClass] = useState('')
  const [senPolarity, setSenPolarity] = useState('')
  // const [modelSelect, setModelSelect] = useState('bert+li')

  const handleClick = () => {
    let data = new FormData()
    data.append('sentence', senClass)
    fetch('http://localhost:8000', {
      method: 'POST',
      body: data,
    })
      .then((res) => res.json())
      .then((res) => {
        setSenPolarity(res?.output ?? '')
      })
  }

  return (
    <ChakraProvider>
      <Container maxW="6xl" my={4}>
        <VStack align="flex-start">
          <Text fontWeight="bold" fontSize="2xl">
            Sentiment Classification
          </Text>
          <Textarea
            value={senClass}
            onChange={(e) => setSenClass(e.target.value)}
            placeholder="Enter sentences here..."
            resize="none"
          />
          <HStack w="full" justify="space-between">
            {/* <RadioGroup onChange={setModelSelect} value={modelSelect}>
              <HStack spacing={6}>
                <Radio value="bert+li">BERT + LI</Radio>
                <Radio value="xlmr+li">XLM-R + LI</Radio>
                <Radio value="gpt2+li">GPT2 + LI</Radio>
              </HStack>
            </RadioGroup> */}
            <Button colorScheme="teal" onClick={handleClick}>
              Run
            </Button>
          </HStack>
        </VStack>
        {senPolarity}
        <Divider my={10} />
        {/* <VStack align="flex-start">
          <Text fontWeight="bold" fontSize="2xl">
            Sentiment Classification
          </Text>
          <Textarea
            value={senSen}
            onChange={(e) => setSenSen(e.target.value)}
            placeholder="Enter sentences here..."
            resize="none"
          />
          <HStack w="full" justify="space-between">
            <Button colorScheme="teal" onClick={handleClick}>
              Run
            </Button>
          </HStack>
        </VStack> */}
      </Container>
    </ChakraProvider>
  )
}

export default App
