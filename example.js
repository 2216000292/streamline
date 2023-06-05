const data = [
    {"0":[1,2,3,4],"1":[1,2,3,4]},
    {"0":[2,3,4,5],"1":[2,3,4,5]},
    {"0":[3,4,5,6],"1":[3,4,5,6]},
    {"0":[4,5,6,7],"1":[4,5,6,7]}
  ];

function process(data) {
    const averages = data.map((item, index) => {
      let total = 0;
      let count = 0;
  
      for(let key in item) {
        const min = Math.min(...item[key]);
        total += min;
        count++;
      }
  
      return { "x": index, "y": total / count, "c": 0 };
    });
  
    return averages;
  }
  
  const result = process(data);
  
  console.log(result);
  