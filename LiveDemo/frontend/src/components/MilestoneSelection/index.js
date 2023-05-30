import { useEffect, useState } from 'react'
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import Select from '@mui/material/Select';

const MileStoneSelection = ({ setMilestone }) => {

    const [milestones, setMilestones] = useState([])

    useEffect(() => {
        async function fetchData() {
            const res = await fetchMilestones()
            console.log("milestoneselection ", res)
            setMilestones(res)
            setMilestone(res[0])
        }

        fetchData()
    }, [])
    
    const fetchMilestones = async () => {
        const res = await fetch("http://localhost:5000/api/hello")
        return res.json()
    }

    const handleChange = e => {

    }

    return (
        <FormControl variant="standard" sx={{ m: 1, minWidth: 120 }}>
            <InputLabel id="milestoneselect">Milestone</InputLabel>
            <Select
                labelId="milestoneselect"
                id="milestoneselect"
                value={milestones[0]}
                onChange={handleChange}
                label="Milestone"
            >
                {milestones.map((ms, idx) => 
                    <MenuItem key={idx} value={ms}>{ms}</MenuItem>
                )}
            </Select>
        </FormControl>
    )
}

export default MileStoneSelection