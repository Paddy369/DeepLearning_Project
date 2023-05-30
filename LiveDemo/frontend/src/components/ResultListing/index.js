import { Card, CardMedia, CardContent } from '@mui/material'
import styles from './styles.module.css'

const ResultListing = ({ className, percentage }) => {

    return (
        <Card sx={{ display: 'flex', width: 250 }} className={styles.card}>
            <CardMedia
                component="img"
                sx={{ width: 55 }}
                image={`/card_thumbnails/cat.jpeg`}
                alt="Cat"
            />
            <CardContent sx={{ flex: '1 0 auto' }}>
                <h3>{className}</h3>
                <div>Percentage: <b>{percentage}</b></div>
            </CardContent>
        </Card>
    )

}

export default ResultListing