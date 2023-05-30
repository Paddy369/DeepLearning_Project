import { useEffect, useState } from 'react'
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import Select from '@mui/material/Select';

const MileStoneSelection = ({ milestone, setMilestone }) => {

    const [milestones, setMilestones] = useState([])

    useEffect(() => {
        async function fetchData() {
            const res = await fetchMilestones()
            console.log("milestoneselection ", res)
            setMilestones(res)
        }

        fetchData()
    }, [])
    
    const fetchMilestones = async () => {
        const res = await fetch("http://localhost:5000/api/milestones")
        return res.json()
    }

    const handleChange = e => {
        setMilestone(e.target.value)
    }

    return (
        <FormControl variant="standard" sx={{ m: 1, minWidth: 120 }}>
            <InputLabel id="milestoneselectlabel">Milestone</InputLabel>
            <Select
                labelId="milestoneselect"
                id="milestoneselect"
                value={milestone}
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