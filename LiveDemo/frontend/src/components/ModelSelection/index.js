import { useEffect, useState } from 'react'
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import Select from '@mui/material/Select';

const ModelSelection = ({ milestone, model, setModel }) => {

    const [models, setModels] = useState([])

    useEffect(() => {
        async function fetchData() {
            const res = await fetchModels()
            setModels(res)
            if(res.length > 0) 
                setModel(res[0])
        }

        fetchData()
    }, [milestone])
    
    const fetchModels = async () => {
        const res = await fetch(`http://localhost:5000/api/models?milestone=${milestone}`)
        return res.json()
    }

    const handleChange = e => {
        setModel(e.target.value)
    }

    return (
        <FormControl variant="standard" sx={{ m: 1, minWidth: 120 }}>
            <InputLabel id="modelselectionlabel">Model</InputLabel>
            <Select
                labelId="modelselect"
                id="modelselect"
                value={model}
                onChange={handleChange}
                label="Model"
            >
                {models.map((ms, idx) => 
                    <MenuItem key={idx} value={ms}>{ms}</MenuItem>
                )}
            </Select>
        </FormControl>
    )
}

export default ModelSelection