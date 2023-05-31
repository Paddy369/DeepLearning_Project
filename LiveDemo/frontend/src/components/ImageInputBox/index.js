import styles from './styles.module.css'
import { useState } from 'react'
import { Typography } from '@mui/material'

const ImageInputBox = ({ setImage, setState }) => {

    const [text, setText] = useState("Drop the Image here")
    const [path, setPath] = useState("")
    const [border, setBorder] = useState(true)

    const handleDrop = e => {
        e.preventDefault()
        const image = e.dataTransfer.files[0]
        setState("loading")
        setText("")
        setPath("./img/" + image.name)
        setImage(image.name)
        setBorder(false)
    }

    const handleDragOver = e => {
        e.preventDefault()
    }

    return (
        <Typography 
          variant="h3" 
          onDragOver={handleDragOver} 
          onDrop={handleDrop} 
          className={`${styles.box} ${border && styles.border}`} 
          style={{backgroundImage: `url(${path})` }}>
            {text}
        </Typography>
    )
}

export default ImageInputBox