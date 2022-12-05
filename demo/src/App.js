import {
  Button,
  ChakraProvider,
  Container,
  HStack,
  Radio,
  RadioGroup,
  Text,
  Textarea,
  VStack,
} from '@chakra-ui/react'
import { useState } from 'react'

const App = () => {
  const [modelSelect, setModelSelect] = useState('bert+li')

  const handleClick = () => {}

  return (
    <ChakraProvider>
      <Container maxW="6xl" my={4}>
        <VStack align="flex-start">
          <Text fontWeight="bold" fontSize="2xl">
            Sentiment Classification
          </Text>
          <Textarea placeholder="Enter sentences here..." resize="none" />
          <HStack w="full" justify="space-between">
            <RadioGroup onChange={setModelSelect} value={modelSelect}>
              <HStack spacing={6}>
                <Radio value="bert+li">BERT + LI</Radio>
                <Radio value="xlmr+li">XLM-R + LI</Radio>
                <Radio value="gpt2+li">GPT2 + LI</Radio>
              </HStack>
            </RadioGroup>
            <Button colorScheme="teal" onClick={handleClick}>
              Run
            </Button>
          </HStack>
        </VStack>
      </Container>
    </ChakraProvider>
  )
}

export default App
