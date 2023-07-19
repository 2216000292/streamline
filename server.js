const express = require('express');
const path = require('path');
const app = express();
const port = 3000;

app.use(express.static('vega_project'));

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'vega_project', 'test.html'));
});

app.listen(port, () => {
  console.log(`Server listening at http://localhost:${port}`);
});

